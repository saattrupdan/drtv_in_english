"""Subtitle generation module for creating WebVTT files.

This module provides functions to convert a list of ``Transcription`` models
into a WebVTT (``.vtt``) subtitle file. Word-level transcriptions are grouped
into readable subtitle cues by splitting on sentence-ending punctuation or a
maximum word count, whichever comes first.
"""

import collections.abc as c
import pathlib as pl
from pathlib import Path

from .logging_config import logger
from .transcribing import Transcription

_SENTENCE_ENDINGS = frozenset(".?!")


def _group_transcriptions(
    transcriptions: list[Transcription], max_words_per_group: int
) -> list[list[Transcription]]:
    """Split word-level transcriptions into subtitle groups.

    A new group is started after a word whose text ends with sentence-ending
    punctuation (``.``, ``?``, ``!``) or when the current group reaches
    ``max_words_per_group`` words.

    Args:
        transcriptions:
            Word-level transcriptions to group.
        max_words_per_group:
            Maximum number of words allowed in a single group.

    Returns:
        A list of groups, where each group is a non-empty list of
        consecutive ``Transcription`` objects.
    """
    groups: list[list[Transcription]] = []
    current: list[Transcription] = []

    for t in transcriptions:
        current.append(t)
        if (
            t.text.rstrip()[-1:] in _SENTENCE_ENDINGS
            or len(current) >= max_words_per_group
        ):
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def generate_subtitles(
    transcriptions: list[Transcription],
    audio_path: str | pl.Path,
    max_words_per_group: int = 8,
) -> c.Generator[tuple[int, int], None, Path]:
    """Generate a WebVTT subtitle file from word-level transcriptions.

    Words are grouped into subtitle cues by splitting on sentence-ending
    punctuation or when ``max_words_per_group`` is reached. Each cue spans
    from the first word's ``start_time`` to the last word's ``end_time``.

    Yields ``(current_index, total)`` progress tuples as each group is
    written.

    Args:
        transcriptions:
            List of ``Transcription`` models (one per word) to display.
        audio_path:
            Path to the source audio file. The output ``.vtt`` file will be
            written to the same directory with the same base name.
        max_words_per_group:
            Maximum number of words in a single subtitle cue.

    Returns:
        The ``Path`` to the generated ``.vtt`` file.

    Yields:
        A tuple of ``(current_index, total)`` progress markers for each
        group processed, where ``current_index`` is 1-based.

    Raises:
        ValueError:
            If the transcriptions list is empty.
        FileNotFoundError:
            If the audio path does not exist.
    """
    audio_path = Path(audio_path)

    if not transcriptions:
        logger.error("Cannot generate subtitles: transcriptions list is empty")
        raise ValueError("Transcriptions list must not be empty")

    if not audio_path.exists():
        logger.error(f"Audio path does not exist: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path = audio_path.with_suffix(suffix=".vtt")
    logger.info(
        f"Generating subtitles for {len(transcriptions)} segments -> {output_path}"
    )

    groups = _group_transcriptions(transcriptions, max_words_per_group)
    total = len(groups)

    vtt_lines: list[str] = ["WEBVTT", ""]

    for index, group in enumerate(groups, start=1):
        start_ts = _format_vtt_timestamp(seconds=group[0].start_time)
        end_ts = _format_vtt_timestamp(seconds=group[-1].end_time)
        text = " ".join(_escape_vtt_text(text=t.text) for t in group)
        vtt_lines.extend([str(index), f"{start_ts} --> {end_ts}", text, ""])
        yield (index, total)

    vtt_content = "\n".join(vtt_lines)

    output_path.write_text(data=vtt_content, encoding="utf-8")

    logger.info(f"Subtitles written to {output_path}")
    return output_path


def _format_vtt_timestamp(seconds: float) -> str:
    """Format a float number of seconds into a WebVTT timestamp.

    Converts seconds into the ``HH:MM:SS.mmm`` format required by the
    WebVTT specification, rounding milliseconds to the nearest integer.

    Args:
        seconds:
            Time in seconds, which may include fractional milliseconds.

    Returns:
        A string in ``HH:MM:SS.mmm`` format, e.g. ``01:23:45.678``.
    """
    total_ms = round(seconds * 1000)
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1_000
    ms = remainder % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _escape_vtt_text(text: str) -> str:
    """Escape special characters for WebVTT cue text.

    Replaces ``<``, ``>``, and ``&`` with their HTML entity equivalents
    to prevent them from being interpreted as markup in the VTT file.

    Args:
        text:
            The raw transcribed text to escape.

    Returns:
        The text with ``<``, ``>``, and ``&`` replaced by ``&lt;``,
        ``&gt;``, and ``&amp;`` respectively.
    """
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text
