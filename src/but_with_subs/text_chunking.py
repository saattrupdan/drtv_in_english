"""Text chunking functionality for splitting transcriptions into segments."""

import re
import string

import nltk
from punctfix import PunctFixer

from .data_models import Transcription
from .logging_config import logger

nltk.download("punkt", quiet=True)


def chunk_transcriptions(
    transcriptions: list[Transcription],
    max_words: int,
    punctuation_model: PunctFixer | None = None,
) -> list[Transcription]:
    """Split transcriptions into segments.

    Args:
        transcriptions:
            A list of Transcription objects to be chunked.
        max_words:
            The maximum number of words per segment.
        punctuation_model (optional):
            The punctuation model to use for fixing punctuation.
            Defaults to a fresh ``PunctFixer`` instance.

    Returns:
        A list of Transcription objects, each containing a segment of the original
        transcription.
    """
    if punctuation_model is None:
        punctuation_model = PunctFixer()

    text = " ".join([transcription.text for transcription in transcriptions])
    text = punctuation_model.punctuate(text=text)

    result: list[Transcription] = []
    for segment in _split_text(text=text, max_words=max_words):
        segment_without_punctuation = re.sub(
            rf"[{string.punctuation}]", "", segment
        ).strip()
        if segment_without_punctuation == "":
            continue

        first_word = segment_without_punctuation.split()[0].lower()
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

        last_word = segment_without_punctuation.split(" ")[-1].lower()
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


def _split_text(*, text: str, max_words: int) -> list[str]:
    """Split text into smaller segments if they exceed max_words.

    Args:
        text:
            The text to split.
        max_words:
            The maximum number of words per segment.

    Returns:
        A list of text segments, each with at most max_words words.
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
        punctuation_segments.extend(segment.split(",;:-"))

    # Fall back to word segmentation
    word_segments: list[str] = list()
    for segment in punctuation_segments:
        if len(segment.split()) <= max_words:
            word_segments.append(segment)
            continue
        words = nltk.word_tokenize(text=segment, language="danish")
        word_segments.extend(
            [
                " ".join(words[i : i + max_words])
                for i in range(0, len(words), max_words)
            ]
        )

    return word_segments
