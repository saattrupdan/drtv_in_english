"""Translation module for translating subtitles between languages.

Uses transformer models for high-quality translation, with support for
batch processing to optimize quality and throughput.

Prioritizes translation quality by processing chunks in batches rather
than one at a time, as batch processing provides more context to the model.
"""

import logging
import re
from pathlib import Path

import bits_and_bobs as bnb
import numpy as np
from transformers import pipeline

from .data_models import Chunk
from .logging_config import logger

# Default model for translation - small100 supports 100+ languages
DEFAULT_TRANSLATION_MODEL = "alirezamsh/small100"


class Translator:
    """High-quality translator using transformer models."""
    
    def __init__(
        self,
        model_id: str = DEFAULT_TRANSLATION_MODEL,
        device: int | None = None,
    ):
        """Initialize the translator.
        
        Args:
            model_id: HuggingFace model ID for translation.
            device: Device to run model on (-1 for CPU, 0+ for GPU).
                   If None, automatically selects best available device.
        """
        logger.info(f"Loading translation model: {model_id}")
        
        with bnb.no_terminal_output():
            self._pipeline = pipeline(
                task="translation",
                model=model_id,
                device=device if device is not None else -1,
            )
        
        logger.info(f"Translation model loaded: {model_id}")
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text segment.
        
        Args:
            text: Text to translate.
            source_lang: Source language code (e.g., "dan" for Danish).
            target_lang: Target language code (e.g., "eng" for English).
            
        Returns:
            Translated text.
        """
        try:
            result = self._pipeline(text, src_lang=source_lang, tgt_lang=target_lang)
            return result[0]["translation_text"]
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
            chunks: List of chunks with text to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            batch_size: Number of texts to translate in parallel.
            
        Returns:
            New list of chunks with translated text.
        """
        if not chunks:
            return []
        
        chunks_with_text = [c for c in chunks if c.text]
        if not chunks_with_text:
            logger.warning("No chunks with text to translate")
            return chunks
        
        logger.info(
            f"Translating {len(chunks_with_text)} chunks "
            f"from {source_lang} to {target_lang}"
        )
        
        texts = [c.text for c in chunks_with_text]
        translated_texts: list[str] = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_results = self._pipeline(
                    batch,
                    src_lang=source_lang,
                    tgt_lang=target_lang,
                )
                translated_texts.extend([r["translation_text"] for r in batch_results])
            except Exception as e:
                logger.error(f"Batch translation failed: {e}")
                for text in batch:
                    try:
                        result = self._pipeline(text, src_lang=source_lang, tgt_lang=target_lang)
                        translated_texts.append(result[0]["translation_text"])
                    except Exception as e2:
                        logger.error(f"Individual translation failed: {e2}")
                        translated_texts.append(text)
        
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
        input_path: Path to input .vtt subtitle file.
        output_path: Path for output .vtt file. If None, uses input path
                    with "_translated" suffix before extension.
        source_lang: Source language code (default: "dan" for Danish).
        target_lang: Target language code (default: "eng" for English).
        model_id: HuggingFace model ID for translation.
        
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
        chunks=chunks,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    
    _write_vtt_file(translated_chunks, output_path)
    
    logger.info(f"Translated subtitles written to {output_path}")
    return output_path


def _parse_vtt_file(path: Path) -> list[Chunk]:
    """Parse a WebVTT file into Chunk objects.
    
    Args:
        path: Path to .vtt file.
        
    Returns:
        List of Chunk objects.
    """
    chunks: list[Chunk] = []
    
    with path.open(encoding="utf-8") as f:
        content = f.read()
    
    cue_pattern = re.compile(
        r'(\d+)\s*\n'
        r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n'
        r'(.*?)\n\n',
        re.DOTALL
    )
    
    for match in cue_pattern.finditer(content):
        start_time = _parse_vtt_timestamp(match.group(2))
        end_time = _parse_vtt_timestamp(match.group(3))
        text = match.group(4).strip()
        
        speaker_match = re.match(r'\(([^)]+)\)\s*', text)
        speaker = None
        if speaker_match:
            speaker = speaker_match.group(1)
            text = text[speaker_match.end():]
        
        text = re.sub(r'<[^>]+>', '', text)
        
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
        chunks: List of Chunk objects.
        path: Output file path.
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
        timestamp: Timestamp string in HH:MM:SS.mmm format.
        
    Returns:
        Time in seconds.
    """
    h, m, s = timestamp.split(":")
    s, ms = s.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _format_vtt_timestamp(seconds: float) -> str:
    """Format seconds into WebVTT HH:MM:SS.mmm timestamp."""
    total_ms = round(seconds * 1000)
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1_000
    ms = remainder % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"