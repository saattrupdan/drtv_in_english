"""Subtitle generation module for creating WebVTT files."""

import collections.abc as c
import pathlib as pl
from pathlib import Path

from .data_models import Transcription
from .logging_config import logger


def generate_subtitles(
    transcriptions: list[Transcription], audio_path: str | pl.Path
) -> c.Generator[tuple[int, int], None, Path]:
    """Generate a WebVTT subtitle file from word-level transcriptions.

    Yields ``(current_index, total)`` progress tuples as each group is
    written.

    Args:
        transcriptions:
            List of ``Transcription`` models (one per word) to display.
        audio_path:
            Path to the source audio file. The output ``.vtt`` file will be
            written to the same directory with the same base name.

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

    total = len(transcriptions)

    with output_path.open(mode="a", encoding="utf-8") as f:
        for index, transcription in enumerate(transcriptions, start=1):
            start_ts = _format_vtt_timestamp(seconds=transcription.start_time)
            end_ts = _format_vtt_timestamp(seconds=transcription.end_time)
            text = _escape_vtt_text(text=transcription.text)
            f.writelines([str(index), f"{start_ts} --> {end_ts}", text, ""])
            yield (index, total)

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
