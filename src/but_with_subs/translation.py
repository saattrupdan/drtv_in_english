"""Translation module for translating subtitles between languages.

Uses transformer models for high-quality translation, with support for
batch processing to optimise quality and throughput.

Prioritises translation quality by processing chunks in batches rather
than one at a time, as batch processing provides more context to the model.
"""

import re
from pathlib import Path

import bits_and_bobs as bnb
import numpy as np
from transformers import M2M100ForConditionalGeneration

from .data_models import Chunk
from .device import get_device
from .logging_config import logger
from .tokenization_small100 import SMALL100Tokenizer

# Default model for translation - small100 supports 100+ languages
DEFAULT_TRANSLATION_MODEL = "alirezamsh/small100"


class Translator:
    """High-quality translator using transformer models."""

    def __init__(self, model_id: str = DEFAULT_TRANSLATION_MODEL) -> None:
        """Initialize the translator.

        Args:
            model_id:
                HuggingFace model ID for translation.
        """
        logger.info(f"Loading translation model: {model_id}")
        self.device = get_device()
        with bnb.no_terminal_output():
            self._model = M2M100ForConditionalGeneration.from_pretrained(model_id)
            self._tokenizer = SMALL100Tokenizer.from_pretrained(model_id)
            self._model = self._model.to(
                self.device  # ty: ignore[invalid-argument-type]
            )

        logger.info(f"Translation model loaded: {model_id}")

    def _translate_batch(self, texts: list[str], tgt_lang: str) -> list[str]:
        """Translate a batch of texts.

        Args:
            texts:
                List of source texts to translate.
            tgt_lang:
                Target language code.

        Returns:
            List of translated texts.
        """
        self._tokenizer.tgt_lang = tgt_lang
        if len(texts) == 1:
            encoded = self._tokenizer(texts[0], return_tensors="pt").to(
                self._model.device
            )
            generated = self._model.generate(  # ty: ignore[invalid-argument-type]
                **encoded
            )
            return [
                self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            ]

        encoded = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self._model.device)
        generated = self._model.generate(**encoded)  # ty: ignore[invalid-argument-type]
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text segment.

        Args:
            text:
                Text to translate.
            source_lang:
                Source language code (e.g., "dan" for Danish).
            target_lang:
                Target language code (e.g., "eng" for English).

        Returns:
            Translated text.
        """
        try:
            return self._translate_batch([text], target_lang)[0]
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

    def translate_chunks(
        self,
        chunks: list[Chunk],
        source_lang: str,
        target_lang: str,
        batch_size: int = 16,
    ) -> list[Chunk]:
        """Translate multiple chunks with batch processing for quality.

        Batch processing improves translation quality by providing more
        context to the model compared to chunk-by-chunk translation.

        Args:
            chunks:
                List of chunks with text to translate.
            source_lang:
                Source language code.
            target_lang:
                Target language code.
            batch_size (optional):
                Number of texts to translate in parallel.

        Returns:
            New list of chunks with translated text.
        """
        if not chunks:
            return []

        chunks_with_text: list[Chunk] = [c for c in chunks if c.text is not None]
        if not chunks_with_text:
            logger.warning("No chunks with text to translate")
            return chunks

        logger.info(
            f"Translating {len(chunks_with_text)} chunks "
            f"from {source_lang} to {target_lang}"
        )

        texts: list[str] = [c.text or "" for c in chunks_with_text]
        translated_texts: list[str] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                translated_texts.extend(self._translate_batch(batch, target_lang))
            except Exception as e:
                logger.error(f"Batch translation failed: {e}")
                for segment_text in batch:
                    try:
                        translated_texts.append(
                            self._translate_batch([segment_text], target_lang)[0]
                        )
                    except Exception as e2:
                        logger.error(f"Individual translation failed: {e2}")
                        translated_texts.append(segment_text)

        translated_chunks: list[Chunk] = []
        text_idx = 0
        for chunk in chunks:
            if chunk.text:
                translated_chunks.append(
                    Chunk(
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        audio=chunk.audio,
                        text=translated_texts[text_idx],
                        speaker=chunk.speaker,
                    )
                )
                text_idx += 1
            else:
                translated_chunks.append(chunk)

        return translated_chunks


def translate_subtitles(
    input_path: Path | str,
    output_path: Path | str | None = None,
    source_lang: str = "dan",
    target_lang: str = "eng",
    model_id: str = DEFAULT_TRANSLATION_MODEL,
) -> Path:
    """Translate a subtitle file from one language to another.

    Args:
        input_path:
            Path to input .vtt subtitle file.
        output_path (optional):
            Path for output .vtt file. If None, uses input path
            with "_translated" suffix before extension.
        source_lang (optional):
            Source language code. Defaults to "dan".
        target_lang (optional):
            Target language code. Defaults to "eng".
        model_id (optional):
            HuggingFace model ID for translation.

    Returns:
        Path to the translated subtitle file.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_translated")
    else:
        output_path = Path(output_path)

    logger.info(f"Translating subtitles: {input_path} -> {output_path}")

    chunks = _parse_vtt_file(input_path)

    translator = Translator(model_id=model_id)
    translated_chunks = translator.translate_chunks(
        chunks=chunks, source_lang=source_lang, target_lang=target_lang
    )

    _write_vtt_file(translated_chunks, output_path)

    logger.info(f"Translated subtitles written to {output_path}")
    return output_path


def _parse_vtt_file(path: Path) -> list[Chunk]:
    """Parse a WebVTT file into Chunk objects.

    Args:
        path:
            Path to .vtt file.

    Returns:
        List of Chunk objects.
    """
    chunks: list[Chunk] = []

    with path.open(encoding="utf-8") as f:
        content = f.read()

    # Pattern to match VTT cues with optional speaker in <v Speaker> format or (Speaker)
    # format
    cue_pattern = re.compile(
        r"(\d+)\s*(?:\(([^)]+)\))?\s*\n"  # Cue number and optional (Speaker) format
        r"(?:<v ([^>]+)>\n)?"  # Optional speaker line in <v Speaker> format
        r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*(?:[A-Za-z]+:[^\n]*)?\n"
        r"((?:(?!\n\n|\n\d+\s*\n(?:<v [^>]+>\n)?\d{2}:\d{2}:\d{2}\.\d{3}).)*)",
        re.DOTALL,
    )

    for match in cue_pattern.finditer(content):
        start_time = _parse_vtt_timestamp(match.group(4))
        end_time = _parse_vtt_timestamp(match.group(5))
        text = match.group(6).strip()

        # Extract speaker from (Speaker) format (group 2) or <v Speaker> format (group
        # 3)
        speaker = match.group(2)  # (Speaker) format
        if speaker is None:
            speaker = match.group(3)  # <v Speaker> format
        if speaker is None:
            speaker_match = re.match(r"\(([^)]+)\)\s*", text)
            if speaker_match:
                speaker = speaker_match.group(1)
                text = text[speaker_match.end() :]

        text = re.sub(r"<[^>]+>", "", text)

        duration = end_time - start_time
        audio = np.zeros(int(duration * 16000), dtype=np.float32)

        chunks.append(
            Chunk(
                start_time=start_time,
                end_time=end_time,
                audio=audio,
                text=text,
                speaker=speaker,
            )
        )

    return chunks


def _write_vtt_file(chunks: list[Chunk], path: Path) -> None:
    """Write chunks to a WebVTT file.

    Args:
        chunks:
            List of Chunk objects.
        path:
            Output file path.
    """
    with path.open(mode="w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")

        for index, chunk in enumerate(chunks, start=1):
            start_ts = _format_vtt_timestamp(chunk.start_time)
            end_ts = _format_vtt_timestamp(chunk.end_time)
            speaker = chunk.speaker or ""

            f.write(f"{index}\n")
            if speaker:
                f.write(f"<v {speaker}>\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{chunk.text}\n")
            f.write("\n")


def _parse_vtt_timestamp(timestamp: str) -> float:
    """Parse WebVTT timestamp to seconds.

    Args:
        timestamp:
            Timestamp string in HH:MM:SS.mmm format.

    Returns:
        Time in seconds.
    """
    h, m, s = timestamp.split(":")
    s, ms = s.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _format_vtt_timestamp(seconds: float) -> str:
    """Format seconds into WebVTT HH:MM:SS.mmm timestamp.

    Returns:
        Formatted timestamp string.
    """
    total_ms = round(seconds * 1000)
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1_000
    ms = remainder % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
