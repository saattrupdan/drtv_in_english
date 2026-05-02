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

        Transcription(start_time=first_word_time, end_time=last_word_time, text=segment)


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

    words = text.split()
    if len(words) <= max_words:
        return [text]

    # Try sentence tokenization
    try:
        sentences = nltk.sent_tokenize(text, language="danish")
    except Exception:
        sentences = []

    if sentences:
        # Check if all sentences are small enough
        short_sentences = [s for s in sentences if len(s.split()) <= max_words]
        long_sentences = [s for s in sentences if len(s.split()) > max_words]

        if len(short_sentences) == len(sentences):
            return sentences

        # Process only long sentences further
        segments: list[str] = []
        for sentence in sentences:
            if len(sentence.split()) <= max_words:
                segments.append(sentence)
            else:
                # Try splitting on punctuation pauses
                pause_pattern = r"([.,;:!?\s])"
                parts = re.split(pause_pattern, sentence)
                joined_parts = "".join(parts)
                if len(joined_parts.split()) <= max_words:
                    segments.append(joined_parts)
                else:
                    # Chunk by word count
                    sentence_words = sentence.split()
                    word_chunks = [
                        " ".join(sentence_words[i : i + max_words])
                        for i in range(0, len(sentence_words), max_words)
                    ]
                    segments.extend(word_chunks)
        return segments

    # Fallback: chunk by word count
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]
