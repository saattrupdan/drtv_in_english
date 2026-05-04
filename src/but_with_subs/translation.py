"""Translation module for translating text between languages.

Uses transformer models for high-quality translation, with support for
batch processing to optimise quality and throughput.

Prioritises translation quality by processing chunks in batches rather
than one at a time, as batch processing provides more context to the model.
"""

import collections.abc as c
import typing as t

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
) -> c.Generator[Chunk, None, None]:
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
        Translated chunks.
    """
    if not chunks:
        return

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
        return

    logger.info(f"Translating {len(chunks_with_text)} chunks to {target_lang}")

    num_batches = len(chunks_with_text) // batch_size
    if len(chunks_with_text) % batch_size != 0:
        num_batches += 1

    for i in range(0, len(chunks_with_text), batch_size):
        batch = chunks_with_text[i : i + batch_size]
        translated_texts = _translate_batch(
            model, tokenizer, [chunk.text or "" for chunk in batch]
        )
        for chunk, translated_text in zip(batch, translated_texts):
            chunk.text = translated_text
            yield chunk


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
        sequences = getattr(generated, "sequences", generated)
        return [
            tokenizer.batch_decode(
                t.cast(t.Any, sequences), skip_special_tokens=True
            )[0]
        ]

    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    generated = model.generate(**encoded)  # ty: ignore[invalid-argument-type]
    sequences = getattr(generated, "sequences", generated)
    return tokenizer.batch_decode(
        t.cast(t.Any, sequences), skip_special_tokens=True
    )


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
    sequences = getattr(generated, "sequences", generated)
    return tokenizer.batch_decode(
        t.cast(t.Any, sequences), skip_special_tokens=True
    )[0]
