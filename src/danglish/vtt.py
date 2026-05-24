"""WebVTT file parsing and writing utilities."""

import re
from pathlib import Path

from .data_models import Chunk


def parse_external_vtt(path: Path) -> list[Chunk]:
    """Parse a standard WebVTT file into Chunk objects.

    Returns:
        List of Chunk objects, one per cue.
    """
    return parse_vtt_text(path.read_text(encoding="utf-8"))


def parse_vtt_text(content: str) -> list[Chunk]:
    """Parse a WebVTT document into Chunk objects.

    Accepts cues without numeric identifiers, with multi-line text and
    leading whitespace used for visual centring (as produced by DR's
    broadcaster subtitles).

    Returns:
        List of Chunk objects, one per cue.
    """
    chunks: list[Chunk] = []

    cue_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})[^\n]*\n"
        r"((?:.+\n?)+?)(?=\n\s*\n|\n[^\n]*-->|\Z)",
        re.MULTILINE,
    )

    for match in cue_pattern.finditer(content):
        start_time = parse_vtt_timestamp(match.group(1))
        end_time = parse_vtt_timestamp(match.group(2))
        raw_text = match.group(3)

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        text = " ".join(lines)
        text = re.sub(r"<[^>]+>", "", text).strip()
        if not text:
            continue

        chunks.append(
            Chunk(start_time=start_time, end_time=end_time, text=text, speaker=None)
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
