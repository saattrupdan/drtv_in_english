"""Subtitle generation module for creating WebVTT files."""

from pathlib import Path
from typing import Generator

from .data_models import Transcription
from .logging_config import logger


def _group_transcriptions(
    transcriptions: list[Transcription], max_words_per_group: int = 8
) -> list[list[Transcription]]:
    """Group transcriptions by sentence boundaries and max word count.

    Splits groups on sentence-ending punctuation (``.``, ``!``, ``?``) and
    also enforces a maximum number of words per group.

    Args:
        transcriptions:
            List of ``Transcription`` models to group.
        max_words_per_group (optional):
            Maximum number of words allowed in a single group. Defaults to 8.

    Returns:
        A list of groups, where each group is a list of ``Transcription``
        models.
    """
    if not transcriptions:
        return []

    groups: list[list[Transcription]] = []
    current_group: list[Transcription] = []

    for transcription in transcriptions:
        current_group.append(transcription)

        # Sentence-ending punctuation splits the group
        if transcription.text and transcription.text[-1] in ".!?":
            groups.append(current_group)
            current_group = []

        # Max words per group also splits
        if len(current_group) >= max_words_per_group:
            groups.append(current_group)
            current_group = []

    if current_group:
        groups.append(current_group)

    return groups


def generate_subtitles(
    transcriptions: list[Transcription],
    audio_path: str | Path,
    max_words_per_group: int = 8,
) -> Generator[tuple[int, int], None, Path]:
    """Generate a WebVTT subtitle file from word-level transcriptions.

    Groups transcriptions into sentence-level cues based on punctuation and
    word count, then writes each group as a single WebVTT cue block.

    Yields ``(current_index, total)`` progress tuples as each group is
    written.

    Args:
        transcriptions:
            List of ``Transcription`` models (one per word) to display.
        audio_path:
            Path to the source audio file. The output ``.vtt`` file will be
            written to the same directory with the same base name.
        max_words_per_group (optional):
            Maximum number of words allowed in a single group. Defaults to 8.

    Yields:
        Progress tuples ``(current_index, total)``.

    Returns:
        The ``Path`` to the generated ``.vtt`` file.

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

    groups = _group_transcriptions(
        transcriptions=transcriptions, max_words_per_group=max_words_per_group
    )
    total = len(groups)

    with output_path.open(mode="w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")

        for index, group in enumerate(groups, start=1):
            start_ts = _format_vtt_timestamp(seconds=group[0].start_time)
            end_ts = _format_vtt_timestamp(seconds=group[-1].end_time)
            text = " ".join(t.text for t in group)
            escaped_text = _escape_vtt_text(text=text)
            f.writelines(
                [
                    f"{str(index)}\n",
                    f"{start_ts} --> {end_ts}\n",
                    f"{escaped_text}\n",
                    "\n",
                ]
            )
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
