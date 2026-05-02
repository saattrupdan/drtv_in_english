"""Text chunking functionality for splitting transcriptions into segments."""

import re
import string

import nltk
from punctfix import PunctFixer

from .data_models import Transcription
from .logging_config import logger

nltk.download("punkt", quiet=True)


def chunk_transcriptions(
    transcriptions: list[Transcription], max_words: int
) -> list[Transcription]:
    """Split transcriptions into segments.

    Args:
        transcriptions:
            A list of Transcription objects to be chunked.
        max_words:
            The maximum number of words per segment.

    Returns:
        A list of Transcription objects, each containing a segment of the original
        transcription.
    """
    text = " ".join([transcription.text for transcription in transcriptions])
    text = PunctFixer(language="da").punctuate(text=text)

    segments = _split_text(text=text, max_words=max_words)

    result: list[Transcription] = []
    for segment in segments:
        first_word = segment.split(" ")[0].lower().strip(string.punctuation)
        first_word_candidates = [
            transcription
            for transcription in transcriptions
            if transcription.text == first_word
        ]
        if not first_word_candidates:
            logger.warning(
                f"Could not find transcription for {first_word!r}. Skipping."
            )
            continue
        first_word_time = first_word_candidates[0].start_time

        last_word = segment.split(" ")[-1].lower().strip(string.punctuation)
        last_word_candidates = [
            transcription
            for transcription in transcriptions
            if transcription.text == last_word
        ]
        if not last_word_candidates:
            logger.warning(f"Could not find transcription for {last_word!r}. Skipping.")
            continue
        last_word_time = last_word_candidates[0].end_time

        result.append(
            Transcription(
                start_time=first_word_time, end_time=last_word_time, text=segment
            )
        )

    return result


def _split_text(*, text: str, max_words: int, _depth: int = 0) -> list[str]:
    """Split text into smaller segments if they exceed max_words.

    Uses recursion to progressively split long segments until they fit.

    Args:
        text:
            The text to split.
        max_words:
            The maximum number of words per segment.
        _depth:
            Internal recursion depth counter (do not pass).

    Returns:
        A list of text segments, each with at most max_words words.
    """
    if not text:
        return []

    words = text.split()
    if len(words) <= max_words:
        return [text]

    # Strategy 1: Sentence tokenization
    sentences = nltk.sent_tokenize(text=text, language="danish")
    if sentences:
        result = []
        for sentence in sentences:
            result.extend(_split_text(text=sentence, max_words=max_words, _depth=_depth + 1))
        if all(len(s.split()) <= max_words for s in result):
            return result

    # Strategy 2: Split on punctuation pauses
    pause_pattern = re.compile(r"([,;:,\-\!\?\,])")
    parts = re.split(pattern=pause_pattern, string=text)
    result = []
    for part in parts:
        result.extend(_split_text(text=part, max_words=max_words, _depth=_depth + 1))
    if all(len(s.split()) <= max_words for s in result):
        return result

    # Strategy 3: Fallback chunk by word count
    part_words = nltk.word_tokenize(text=text, language="danish")
    return [
        " ".join(part_words[i : i + max_words])
        for i in range(0, len(part_words), max_words)
    ]
