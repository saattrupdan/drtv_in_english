"""Subtitle generation module for creating WebVTT files.

This module provides functions to convert a list of ``Transcription`` models
into a WebVTT (``.vtt``) subtitle file. The main function yields progress
tuples during processing and returns the path to the generated file.
"""

import collections.abc as c
import pathlib as pl
from pathlib import Path

from .logging_config import logger
from .transcribing import Transcription


def generate_subtitles(
    transcriptions: list[Transcription], audio_path: str | pl.Path
) -> c.Generator[tuple[int, int], None, Path]:
    """Generate a WebVTT subtitle file from a list of transcriptions.

    Validates the input, derives the output path from the audio file path,
    and writes each transcription as a VTT cue with properly formatted
    timestamps. Yields ``(current_index, total)`` progress tuples for
    each transcription processed.

    Args:
        transcriptions:
            List of ``Transcription`` models to convert into subtitle cues.
        audio_path:
            Path to the source audio file. The output ``.vtt`` file will be
            written to the same directory with the same base name.

    Returns:
        The ``Path`` to the generated ``.vtt`` file.

    Yields:
        A tuple of ``(current_index, total)`` progress markers for each
        transcription processed, where ``current_index`` is 1-based.

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

    total = len(transcriptions)
    logger.info(f"Generating subtitles for {total} segments -> {output_path}")

    vtt_lines: list[str] = ["WEBVTT", ""]

    for index, transcription in enumerate(transcriptions, start=1):
        cue_number = index
        start_timestamp = _format_vtt_timestamp(seconds=transcription.start_time)
        end_timestamp = _format_vtt_timestamp(seconds=transcription.end_time)
        escaped_text = _escape_vtt_text(text=transcription.text)
        vtt_lines.extend(
            [
                str(cue_number),
                f"{start_timestamp} --> {end_timestamp}",
                escaped_text,
                "",
            ]
        )
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
