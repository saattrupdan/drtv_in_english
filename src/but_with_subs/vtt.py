"""WebVTT file parsing and writing utilities."""

import re
from pathlib import Path

import numpy as np

from .constants import TARGET_SAMPLE_RATE
from .data_models import Chunk


def parse_vtt_file(path: Path) -> list[Chunk]:
    """Parse a WebVTT file into Chunk objects.

    Args:
        path:
            Path to .vtt file.

    Returns:
        List of Chunk objects.
    """
    chunks: list[Chunk] = []

    with path.open(encoding="utf-8") as f:
        content = f.read()

    # Pattern to match VTT cues with optional speaker in <v Speaker> format or (Speaker)
    # format
    cue_pattern = re.compile(
        r"(\d+)\s*(?:\(([^)]+)\))?\s*\n"  # Cue number and optional (Speaker) format
        r"(?:<v ([^>]+)>\n)?"  # Optional speaker line in <v Speaker> format
        r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*(?:[A-Za-z]+:[^\n]*)?\n"
        r"((?:(?!\n\n|\n\d+\s*\n(?:<v [^>]+>\n)?\d{2}:\d{2}:\d{2}\.\d{3}).)*)",
        re.DOTALL,
    )

    for match in cue_pattern.finditer(content):
        start_time = parse_vtt_timestamp(match.group(4))
        end_time = parse_vtt_timestamp(match.group(5))
        text = match.group(6).strip()

        # Extract speaker from (Speaker) format (group 2) or <v Speaker> format (group
        # 3)
        speaker = match.group(2)  # (Speaker) format
        if speaker is None:
            speaker = match.group(3)  # <v Speaker> format
        if speaker is None:
            speaker_match = re.match(r"\(([^)]+)\)\s*", text)
            if speaker_match:
                speaker = speaker_match.group(1)
                text = text[speaker_match.end() :]

        text = re.sub(r"<[^>]+>", "", text)

        duration = end_time - start_time
        audio = np.zeros(int(duration * TARGET_SAMPLE_RATE), dtype=np.float32)

        chunks.append(
            Chunk(
                start_time=start_time,
                end_time=end_time,
                audio=audio,
                text=text,
                speaker=speaker,
            )
        )

    return chunks


def write_vtt_file(chunks: list[Chunk], path: Path) -> None:
    """Write chunks to a WebVTT file.

    Args:
        chunks:
            List of Chunk objects.
        path:
            Output file path.
    """
    with path.open(mode="w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")

        for index, chunk in enumerate(chunks, start=1):
            start_ts = format_vtt_timestamp(chunk.start_time)
            end_ts = format_vtt_timestamp(chunk.end_time)
            speaker = chunk.speaker or ""

            f.write(f"{index}\n")
            if speaker:
                f.write(f"<v {speaker}>\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{chunk.text}\n")
            f.write("\n")


def parse_vtt_timestamp(timestamp: str) -> float:
    """Parse WebVTT timestamp to seconds.

    Args:
        timestamp:
            Timestamp string in HH:MM:SS.mmm format.

    Returns:
        Time in seconds.
    """
    h, m, s = timestamp.split(":")
    s, ms = s.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds into WebVTT HH:MM:SS.mmm timestamp.

    Returns:
        Formatted timestamp string.
    """
    total_ms = round(seconds * 1000)
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1_000
    ms = remainder % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
