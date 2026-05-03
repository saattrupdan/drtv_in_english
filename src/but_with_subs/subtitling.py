"""Subtitle generation module for creating WebVTT files."""

from pathlib import Path
from typing import Any

from .constants import MIN_CHUNK_DISPLAY_LENGTH_SECONDS, OVERLAPPING_SPEAKER_COLORS
from .data_models import Chunk
from .logging_config import logger


def generate_subtitles(chunks: list[Chunk], audio_path: str | Path) -> Path:
    """Generate a WebVTT subtitle file from word-level transcriptions.

    Args:
        chunks: List of chunks to generate subtitles from.
        audio_path: Path to the source audio file. Output `.vtt` file
            will be in the same directory with the same base name.

    Returns:
        Path to the generated subtitle file.

    Raises:
        ValueError: If there are no chunks.
        FileNotFoundError: If the audio path does not exist.
    """
    audio_path = Path(audio_path)

    if not chunks:
        logger.error("Cannot generate subtitles since there are no chunks")
        raise ValueError("Transcriptions list must not be empty")

    if not audio_path.exists():
        logger.error(f"Audio path does not exist: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Sort chunks by start time to ensure chronological order
    chunks = sorted(chunks, key=lambda c: c.start_time)

    output_path = audio_path.with_suffix(suffix=".vtt")
    output_path.unlink(missing_ok=True)

    logger.info(f"Generating subtitles for {len(chunks)} chunks -> {output_path}")

    # Detect overlapping speakers and assign colors
    overlapping_speakers = _detect_overlapping_speakers(chunks)
    speaker_colors = _assign_speaker_colors(overlapping_speakers)

    with output_path.open(mode="a", encoding="utf-8") as f:
        for index, chunk in enumerate(chunks, start=1):
            chunk.end_time = max(
                chunk.end_time, chunk.start_time + MIN_CHUNK_DISPLAY_LENGTH_SECONDS
            )

            start_ts = _format_vtt_timestamp(seconds=chunk.start_time)
            end_ts = _format_vtt_timestamp(seconds=chunk.end_time)
            escaped_text = _escape_vtt_text(text=chunk.text or "")
            speaker = chunk.speaker or "N/A"
            
            # Apply color if speaker has an assigned color (overlapping speaker)
            if chunk.speaker and chunk.speaker in speaker_colors:
                escaped_text = _apply_speaker_color(
                    escaped_text, speaker_colors[chunk.speaker]
                )
            
            f.write(f"{str(index)} ({speaker})\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{escaped_text}\n")
            f.write("\n")

    logger.info(f"Subtitles written to {output_path}")
    return output_path


def _format_vtt_timestamp(seconds: float) -> str:
    """Format seconds into WebVTT ``HH:MM:SS.mmm`` timestamp."""
    total_ms = round(seconds * 1000)
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1_000
    ms = remainder % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _escape_vtt_text(text: str) -> str:
    """Escape ``<``, ``>``, and ``&`` for WebVTT cue text."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _detect_overlapping_speakers(chunks: list[Chunk]) -> dict[str, list[tuple[float, float]]]:
    """Detect time periods where multiple speakers overlap.
    
    Args:
        chunks: List of chunks with speaker information.
        
    Returns:
        Dictionary mapping speaker names to their overlapping time ranges.
        Only includes speakers who have at least one overlap with another speaker.
    """
    # Build list of (start, end, speaker) tuples
    speaker_segments: list[tuple[float, float, str]] = []
    for chunk in chunks:
        if chunk.speaker:
            speaker_segments.append((chunk.start_time, chunk.end_time, chunk.speaker))
    
    if len(speaker_segments) < 2:
        return {}
    
    # Sort by start time
    speaker_segments.sort(key=lambda x: x[0])
    
    # Find overlapping segments
    overlapping_speakers: dict[str, list[tuple[float, float]]] = {}
    
    for i, seg1 in enumerate(speaker_segments):
        start1, end1, speaker1 = seg1
        for seg2 in speaker_segments[i + 1:]:
            start2, end2, speaker2 = seg2
            
            # Check for overlap: seg1.start < seg2.end AND seg2.start < seg1.end
            if start1 < end2 and start2 < end1:
                # Record overlap for both speakers
                overlapping_speakers.setdefault(speaker1, []).append((start1, end1))
                overlapping_speakers.setdefault(speaker2, []).append((start2, end2))
                
                # Stop checking this seg1 once we find an overlap (since sorted)
                break
    
    return overlapping_speakers


def _assign_speaker_colors(
    overlapping_speakers: dict[str, list[tuple[float, float]]]
) -> dict[str, str]:
    """Assign colors to overlapping speakers.
    
    Args:
        overlapping_speakers: Dictionary from _detect_overlapping_speakers().
        
    Returns:
        Dictionary mapping speaker names to their assigned hex colors.
        First speaker gets default (no color), subsequent speakers get colors.
    """
    if not overlapping_speakers:
        return {}
    
    speaker_colors: dict[str, str] = {}
    
    # Sort speakers by first overlap time for consistent coloring
    sorted_speakers = sorted(
        overlapping_speakers.keys(),
        key=lambda s: min(t[0] for t in overlapping_speakers[s])
    )
    
    # First speaker remains uncolored (default black/white)
    # Subsequent speakers get colors from palette
    for i, speaker in enumerate(sorted_speakers):
        if i == 0:
            # First speaker - no color (default)
            continue
        color_index = (i - 1) % len(OVERLAPPING_SPEAKER_COLORS)
        speaker_colors[speaker] = OVERLAPPING_SPEAKER_COLORS[color_index]
    
    return speaker_colors


def _apply_speaker_color(text: str, color_hex: str) -> str:
    """Wrap text with WebVTT color styling.
    
    WebVTT uses the <c.color>text</c> syntax for inline styling.
    
    Args:
        text: The text to color.
        color_hex: Hex color code (e.g., "#E69F00").
        
    Returns:
        Text wrapped with WebVTT color tags.
    """
    return f"<c.{color_hex}>{text}</c>"
