"""Text chunking functionality for splitting transcriptions into segments."""

import difflib
import re
import string
import typing as t

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


def _find_matching_chunks(
    target: str,
    word_chunks: list[Chunk],
    max_distance: int = 2,
) -> list[Chunk]:
    """Find word chunks whose cleaned text closely matches target.

    Uses SequenceMatcher ratio for fuzzy matching.
    Falls back to exact match if no fuzzy match is found.

    Args:
        target: The target word or phrase to match.
        word_chunks: List of candidate word chunks.
        max_distance: Maximum Levenshtein distance for consideration.

    Returns:
        List of matching word chunks, sorted by similarity score descending.
    """
    cleaned_target = re.sub(PUNCTUATION_PATTERN, "", target).strip().lower()
    candidates: list[tuple[float, Chunk]] = []
    for wc in word_chunks:
        if wc.text is None:
            continue
        cleaned_wc = re.sub(PUNCTUATION_PATTERN, "", wc.text).strip().lower()
        ratio = difflib.SequenceMatcher(None, cleaned_target, cleaned_wc).ratio()
        if ratio >= 0.85:  # 85% similarity threshold
            candidates.append((ratio, wc))

    # Sort by similarity descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    if candidates and candidates[0][0] >= 0.95:
        return [candidates[0][1]]  # strong match
    elif candidates:
        logger.debug(
            f"Fuzzy match for {cleaned_target!r}: best={candidates[0][1].text!r} "
            f"(ratio={candidates[0][0]:.2f})"
        )
        return [candidates[0][1]]
    return []


def group_word_chunks(
    word_chunks: list[Chunk], punctuation_model: PunctFixer, max_words: int
) -> list[Chunk]:
    """Group word chunks into segments.

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
    # The punctuation model requires the input to be lower case and without any
    # punctuation
    text = " ".join(
        [
            re.sub(PUNCTUATION_PATTERN, "", word_chunk.text.lower())
            for word_chunk in word_chunks
            if word_chunk.text
        ]
    )
    text = punctuation_model.punctuate(text=text)

    result: list[Chunk] = []
    for segment in _split_text(text=text, max_words=max_words):
        # Strip the punctuation from the segment, only to be able to locate it amongst
        # the word chunks
        segment_without_punctuation = re.sub(PUNCTUATION_PATTERN, "", segment).strip()
        if segment_without_punctuation == "":
            continue

        # Get the starting time of the segment
        first_word = segment_without_punctuation.split()[0].lower()
        first_word_candidates = _find_matching_chunks(first_word, word_chunks)
        if not first_word_candidates:
            logger.warning(
                f"Could not find transcription for {first_word!r}. Skipping."
            )
            continue
        segment_start = first_word_candidates[0].start_time

        # Get the ending time of the segment
        last_word = segment_without_punctuation.split(" ")[-1].lower()
        last_word_candidates = _find_matching_chunks(last_word, word_chunks)
        # Also enforce temporal ordering: must be after segment_start
        last_word_candidates = [c for c in last_word_candidates if c.end_time > segment_start]
        if not last_word_candidates:
            logger.warning(f"Could not find transcription for {last_word!r}. Skipping.")
            continue
        segment_end = last_word_candidates[0].end_time

        if segment_end - segment_start < MIN_CHUNK_LENGTH_SECONDS:
            continue

        # Identify all the word chunks that fall within the segment
        word_chunks_in_segment = [
            word_chunk
            for word_chunk in word_chunks
            if word_chunk.start_time >= segment_start
            and word_chunk.end_time <= segment_end
        ]

        result.append(
            Chunk(
                start_time=segment_start,
                end_time=segment_end,
                audio=np.concatenate(
                    [word_chunk.audio for word_chunk in word_chunks_in_segment], axis=0
                ),
                text=segment,
                speaker=word_chunks_in_segment[0].speaker,
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
            [s.strip() for s in re.split(r'[,;:\-\u2013\u2014]', segment) if s.strip()]
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
