"""Translation module for translating text between languages.

Uses transformer models for high-quality translation, with support for
batch processing to optimise quality and throughput.

Prioritises translation quality by processing chunks in batches rather
than one at a time, as batch processing provides more context to the model.
"""

import collections.abc as c

import bits_and_bobs as bnb
from transformers import M2M100ForConditionalGeneration

from .constants import TRANSLATION_MODEL
from .data_models import Chunk
from .device import get_device
from .logging_config import logger
from .tokenization_small100 import SMALL100Tokenizer


def translate_chunks(
    chunks: list[Chunk],
    target_lang: str,
    batch_size: int,
    model_id: str = TRANSLATION_MODEL,
) -> c.Generator[tuple[int, int], None, list[Chunk]]:
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
        model_id (optional):
            HuggingFace model ID for translation. Defaults to
            ``DEFAULT_TRANSLATION_MODEL``.

    Yields:
        Progress tuples ``(current, total)`` after each batch completes.

    Returns:
        List of translated chunks.
    """
    if not chunks:
        return list()

    logger.info(f"Loading translation model: {model_id}")
    device = get_device()
    with bnb.no_terminal_output():
        model = M2M100ForConditionalGeneration.from_pretrained(model_id)
        tokenizer = SMALL100Tokenizer.from_pretrained(model_id, tgt_lang=target_lang)
        model = model.to(device)  # ty: ignore[invalid-argument-type]

    logger.info(f"Translation model loaded: {model_id}")

    chunks_with_text: list[Chunk] = [c for c in chunks if c.text is not None]
    if not chunks_with_text:
        logger.warning("No chunks with text to translate")
        return list()

    logger.info(f"Translating {len(chunks_with_text)} chunks to {target_lang}")

    texts: list[str] = [c.text or "" for c in chunks_with_text]
    translated_texts: list[str] = []

    num_batches = len(texts) // batch_size
    if len(texts) % batch_size != 0:
        num_batches += 1

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            translated_texts.extend(_translate_batch(model, tokenizer, batch))
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            for segment_text in batch:
                try:
                    translated_texts.append(
                        translate_single(model, tokenizer, segment_text)
                    )
                except Exception as e2:
                    logger.error(f"Individual translation failed: {e2}")
                    translated_texts.append(segment_text)

        yield (i + len(batch), num_batches)

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


def _translate_batch(
    model: M2M100ForConditionalGeneration,
    tokenizer: SMALL100Tokenizer,
    texts: list[str],
) -> list[str]:
    """Translate a batch of texts.

    Args:
        model: The translation model.
        tokenizer: The tokenizer.
        texts: List of source texts to translate.

    Returns:
        List of translated texts.
    """
    if len(texts) == 1:
        encoded = tokenizer(texts[0], return_tensors="pt").to(model.device)
        generated = model.generate(**encoded)  # ty: ignore[invalid-argument-type]
        return [tokenizer.batch_decode(generated, skip_special_tokens=True)[0]]

    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    generated = model.generate(**encoded)  # ty: ignore[invalid-argument-type]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def translate_single(
    model: M2M100ForConditionalGeneration, tokenizer: SMALL100Tokenizer, text: str
) -> str:
    """Translate a single text segment.

    Args:
        model: The translation model.
        tokenizer: The tokenizer.
        text: Text to translate.

    Returns:
        Translated text.
    """
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    generated = model.generate(**encoded)  # ty: ignore[invalid-argument-type]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
