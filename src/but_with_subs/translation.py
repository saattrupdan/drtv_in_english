"""Translation module for translating text between languages.

Uses transformer models for high-quality translation, with support for
batch processing to optimise quality and throughput.

Prioritises translation quality by processing chunks in batches rather
than one at a time, as batch processing provides more context to the model.
"""

import bits_and_bobs as bnb
from transformers import M2M100ForConditionalGeneration

from .data_models import Chunk
from .device import get_device
from .logging_config import logger
from .tokenization_small100 import SMALL100Tokenizer


class Translator:
    """High-quality translator using transformer models."""

    def __init__(self, model_id: str) -> None:
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

    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate a single text segment.

        Args:
            text:
                Text to translate.
            target_lang:
                Target language code (e.g., "en" for English).

        Returns:
            Translated text.
        """
        try:
            return self._translate_batch([text], target_lang)[0]
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

    def translate_chunks(
        self, chunks: list[Chunk], target_lang: str, batch_size: int
    ) -> list[Chunk]:
        """Translate multiple chunks with batch processing for quality.

        Batch processing improves translation quality by providing more
        context to the model compared to chunk-by-chunk translation.

        Args:
            chunks:
                List of chunks with text to translate.
            target_lang:
                Target language code.
            batch_size:
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

        logger.info(f"Translating {len(chunks_with_text)} chunks to {target_lang}")

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
