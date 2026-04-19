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


def main(video_path: str) -> None:
    """Run audio extraction on a video file.

    Args:
        video_path:
            Path to the input video file.
    """
    path = Path(video_path)

    if not path.is_file():
        logger.error(f"File not found: {video_path}")
        sys.exit(1)

    logger.info(f"Extracting audio from {video_path}...")
    output_path = extract_audio(video_path=path)
    logger.info(f"Audio extracted to {output_path}")


@click.command()
@click.argument("video_path", required=True)
def cli(video_path: str) -> None:
    """Extract audio from a video file."""
    main(video_path=video_path)


if __name__ == "__main__":
    cli()
