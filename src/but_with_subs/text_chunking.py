"""Text chunking functionality for splitting transcriptions into segments."""

import re
import string

import nltk
import numpy as np
from punctfix import PunctFixer

from .constants import MIN_CHUNK_LENGTH_SECONDS
from .data_models import Chunk
from .logging_config import logger

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

PUNCTUATION_PATTERN = rf"[{string.punctuation}]"





def group_word_chunks(
    word_chunks: list[Chunk], punctuation_model: PunctFixer, max_words: int
) -> list[Chunk]:
    """Group word chunks into segments.

    Uses a sequential lockstep pointer to align cleaned word chunks with
    punctuated segments. This avoids the global fuzzy-match bug where
    common words (e.g. "jeg", "det", "er") anchored every segment to the
    first occurrence in the audio.

    Args:
        word_chunks:
            A list of Chunk objects to be chunked.
        punctuation_model:
            The punctuation model to use for fixing punctuation.
        max_words:
            The maximum number of words per segment.

    Returns:
        A list of Chunk objects, each containing a segment of the original
        transcription.
    """
    # 1. Build the cleaned word stream that PunctFixer sees, preserving the
    #    1:1 mapping cleaned_words[i] <-> word_chunks[i].
    cleaned_words: list[str] = []
    indexed_chunks: list[Chunk] = []
    for wc in word_chunks:
        if not wc.text:
            continue
        cleaned = re.sub(PUNCTUATION_PATTERN, "", wc.text.lower()).strip()
        if not cleaned:
            continue
        cleaned_words.append(cleaned)
        indexed_chunks.append(wc)

    punctuated = punctuation_model.punctuate(text=" ".join(cleaned_words))

    # 2. Split the punctuated text into readable segments.
    #    PunctFixer preserves word order and count for non-empty inputs, so
    #    iterating segments and consuming N cleaned words per segment is
    #    sufficient.
    result: list[Chunk] = []
    cursor = 0
    for segment in _split_text(text=punctuated, max_words=max_words):
        tokens = [
            re.sub(PUNCTUATION_PATTERN, "", t).strip().lower()
            for t in segment.split()
        ]
        tokens = [t for t in tokens if t]
        if not tokens:
            continue
        n = len(tokens)
        if cursor + n > len(indexed_chunks):
            logger.warning(
                f"Word/segment alignment exhausted at cursor={cursor} for "
                f"segment={segment!r}; skipping."
            )
            break
        window = indexed_chunks[cursor : cursor + n]
        cursor += n
        start = window[0].start_time
        end = window[-1].end_time
        if end - start < MIN_CHUNK_LENGTH_SECONDS:
            continue
        result.append(
            Chunk(
                start_time=start,
                end_time=end,
                audio=np.concatenate([w.audio for w in window], axis=0),
                text=segment,
                speaker=window[0].speaker,
            )
        )
    return result


def _split_text(*, text: str, max_words: int) -> list[str]:
    """Split text into segments of at most max_words words.

    Uses sentence segmentation, punctuation splitting, and word splitting
    as fallback strategies.

    Returns:
        A list of text segments.
    """
    if not text:
        return []

    # Try sentence segmentation first
    sentence_segments: list[str] = list()
    sentences = nltk.sent_tokenize(text=text, language="danish")
    if sentences:
        for sentence in sentences:
            sentence_segments.append(sentence)
    else:
        sentence_segments = [text]

    # Try punctuation segmentation
    punctuation_segments: list[str] = list()
    for segment in sentence_segments:
        if len(segment.split()) <= max_words:
            punctuation_segments.append(segment)
            continue
        # Use regex to split on any of the punctuation characters
        # Handles comma, semicolon, colon, dash, en-dash, em-dash
        punctuation_segments.extend(
            [s.strip() for s in re.split(r"[,;:\-\u2013\u2014]", segment) if s.strip()]
        )

    # Fall back to word segmentation
    word_segments: list[str] = list()
    for segment in punctuation_segments:
        if len(segment.split()) <= max_words:
            word_segments.append(segment)
            continue
        words = segment.split()
        word_segments.extend(
            [
                " ".join(words[i : i + max_words])
                for i in range(0, len(words), max_words)
            ]
        )

    return word_segments
