"""Test script to demonstrate the audio chunking function in action.

Usage:
    uv run src/scripts/chunk_audio.py [audio_path]
"""

import sys
from pathlib import Path

import click

from but_with_subs import configure_logging
from but_with_subs.audio_chunking import chunk_by_audio
from but_with_subs.audio_loading import load_audio
from but_with_subs.logging_config import logger

configure_logging()


@click.command()
@click.argument("audio_path", required=True)
def main(audio_path: str) -> None:
    """Run audio chunking on an audio file.

    Loads the audio file, splits it into chunks based on silence breaks,
    and logs the total number of chunks.

    Args:
        audio_path:
            Path to the input audio file.
    """
    path = Path(audio_path)
    if not path.is_file():
        logger.error(f"File not found: {audio_path}")
        sys.exit(1)

    logger.info(f"Chunking audio from {audio_path}...")
    audio = load_audio(path)
    chunks = list(chunk_by_audio(audio=audio))

    logger.info(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
