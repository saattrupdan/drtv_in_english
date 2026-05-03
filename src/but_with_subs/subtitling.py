"""Subtitle generation module for creating WebVTT files."""

from pathlib import Path

from .constants import MIN_CHUNK_DISPLAY_LENGTH_SECONDS
from .data_models import Chunk
from .logging_config import logger


def generate_subtitles(chunks: list[Chunk], audio_path: str | Path) -> Path:
    """Generate a WebVTT subtitle file from word-level transcriptions.

    Args:
        chunks:
            The list of chunks to generate subtitles from. Each chunk corresponds to one
            piece of subtitle.
        audio_path:
            Path to the source audio file. The output ``.vtt`` file will be
            written to the same directory with the same base name.

    Returns:
        The path to the generated subtitle file.

    Raises:
        ValueError:
            If there are no chunks.
        FileNotFoundError:
            If the audio path does not exist.
    """
    audio_path = Path(audio_path)

    if not chunks:
        logger.error("Cannot generate subtitles since there are no chunks")
        raise ValueError("Transcriptions list must not be empty")

    if not audio_path.exists():
        logger.error(f"Audio path does not exist: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path = audio_path.with_suffix(suffix=".vtt")
    output_path.unlink(missing_ok=True)

    logger.info(f"Generating subtitles for {len(chunks)} chunks -> {output_path}")

    with output_path.open(mode="a", encoding="utf-8") as f:
        for index, chunk in enumerate(chunks, start=1):
            chunk.end_time = max(
                chunk.end_time, chunk.start_time + MIN_CHUNK_DISPLAY_LENGTH_SECONDS
            )

            start_ts = _format_vtt_timestamp(seconds=chunk.start_time)
            end_ts = _format_vtt_timestamp(seconds=chunk.end_time)
            escaped_text = _escape_vtt_text(text=chunk.text or "")
            speaker = chunk.speaker or "N/A"
            f.write(f"{str(index)} ({speaker})\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{escaped_text}\n")
            f.write("\n")

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
