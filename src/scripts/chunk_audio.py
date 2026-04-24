"""Test script to demonstrate the audio chunking function in action.

Usage:
    uv run src/scripts/chunk_audio.py [audio_path]
"""

import sys
from pathlib import Path

import click

from but_with_subs.chunking import chunk_audio
from but_with_subs.logging_config import logger


@click.command()
@click.argument("audio_path", required=True)
def main(audio_path: str) -> None:
    """Run audio chunking on an audio file.

    Loads the audio file, splits it into chunks based on silence breaks,
    and prints information about each chunk.

    Args:
        audio_path:
            Path to the input audio file.
    """
    path = Path(audio_path)
    if not path.is_file():
        logger.error(f"File not found: {audio_path}")
        sys.exit(1)

    logger.info(f"Chunking audio from {audio_path}...")
    chunks = list(chunk_audio(audio_path=path))

    logger.info(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        duration = chunk.end_time - chunk.start_time
        logger.info(
            f"Chunk {i}: "
            f"start={chunk.start_time:.3f} s, "
            f"end={chunk.end_time:.3f} s, "
            f"duration={duration:.3f} s"
        )


if __name__ == "__main__":
    main()
