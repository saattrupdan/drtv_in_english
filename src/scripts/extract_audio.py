"""Test script to demonstrate the audio extraction function in action.

Usage:
    uv run src/scripts/extract_audio.py [video_path]
"""

import logging
import sys
from pathlib import Path

import click

from but_with_subs.audio_extraction import extract_audio

logger = logging.getLogger(__package__)


@click.command()
@click.argument("video_path", required=True)
def main(video_path: str) -> None:
    """Run audio extraction on a video file.

    Args:
        video_path:
            Path to the input video file.
    """
    path = Path(video_path)
    if not path.is_file():
        logger.error("File not found: %s", video_path)
        sys.exit(1)
    elif path.with_suffix(".wav").exists():
        logger.error(f"Output file already exists: {path.with_suffix('.wav')}")
        sys.exit(1)

    extract_audio(video_path=path)


if __name__ == "__main__":
    main()
