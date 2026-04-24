"""Subtitle generation module for creating WebVTT files.

This module provides functions to convert a list of ``Transcription`` models
into a WebVTT (``.vtt``) subtitle file using a rolling-window display. Words
appear as they are spoken and disappear either when pushed out by newer words
or after a configurable duration, whichever comes first.
"""

import collections.abc as c
import pathlib as pl
from pathlib import Path

from .logging_config import logger
from .transcribing import Transcription

_EXPIRE = 0
_APPEAR = 1


def generate_subtitles(
    transcriptions: list[Transcription],
    audio_path: str | pl.Path,
    max_words: int = 8,
    word_duration: float = 3.0,
) -> c.Generator[tuple[int, int], None, Path]:
    """Generate a WebVTT subtitle file with rolling-window word display.

    Each transcription is treated as a single word. Words appear at their
    ``start_time`` and are removed either when ``max_words`` newer words
    have appeared (pushing old words out) or ``word_duration`` seconds
    after they first appeared, whichever comes first.

    Yields ``(current_index, total)`` progress tuples as each word is
    processed.

    Args:
        transcriptions:
            List of ``Transcription`` models (one per word) to display.
        audio_path:
            Path to the source audio file. The output ``.vtt`` file will be
            written to the same directory with the same base name.
        max_words:
            Maximum number of words visible at the same time.
        word_duration:
            Seconds a word stays visible after it first appears (unless
            pushed out sooner by newer words).

    Returns:
        The ``Path`` to the generated ``.vtt`` file.

    Yields:
        A tuple of ``(current_index, total)`` progress markers for each
        word processed, where ``current_index`` is 1-based.

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

    # Build timeline events sorted so expires are processed before appears
    # at the same timestamp.
    events: list[tuple[float, int, int]] = []
    for i, t in enumerate(transcriptions):
        events.append((t.start_time, _APPEAR, i))
        events.append((t.start_time + word_duration, _EXPIRE, i))
    events.sort()

    # Walk events chronologically, tracking visible words.
    active: list[int] = []
    states: list[tuple[float, list[int]]] = []
    appeared_count = 0
    prev_active: list[int] | None = None

    ev_idx = 0
    while ev_idx < len(events):
        time = events[ev_idx][0]
        # Process all events that share the same timestamp.
        while ev_idx < len(events) and events[ev_idx][0] == time:
            _, event_type, idx = events[ev_idx]
            if event_type == _EXPIRE:
                if idx in active:
                    active.remove(idx)
            else:
                active.append(idx)
                appeared_count += 1
                yield (appeared_count, total)
            ev_idx += 1

        # Enforce the rolling-window limit.
        while len(active) > max_words:
            active.pop(0)

        if active != prev_active:
            states.append((time, list(active)))
            prev_active = list(active)

    # Convert states into VTT cues.
    vtt_lines: list[str] = ["WEBVTT", ""]
    cue_number = 0

    for j, (start_time, indices) in enumerate(states):
        if not indices:
            continue

        if j + 1 < len(states):
            end_time = states[j + 1][0]
        else:
            end_time = max(
                transcriptions[idx].start_time + word_duration for idx in indices
            )

        if end_time <= start_time:
            continue

        cue_number += 1
        text = " ".join(
            _escape_vtt_text(text=transcriptions[idx].text) for idx in indices
        )
        start_ts = _format_vtt_timestamp(seconds=start_time)
        end_ts = _format_vtt_timestamp(seconds=end_time)
        vtt_lines.extend(
            [str(cue_number), f"{start_ts} --> {end_ts}", text, ""]
        )

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
