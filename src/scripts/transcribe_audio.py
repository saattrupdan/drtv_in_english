"""Test script to demonstrate audio transcription with Wav2Vec2.

Usage:
    uv run src/scripts/transcribe_audio.py -a [audio_path]
"""

import logging
import sys
from pathlib import Path

import click
import librosa
from transformers import pipeline

from but_with_subs.chunking import chunk_audio
from but_with_subs.device import get_device
from but_with_subs.transcribing import transcribe

logger = logging.getLogger(__package__)


@click.command()
@click.argument("audio_path", type=str)
def main(audio_path: str) -> None:
    """Transcribe an audio file using Wav2Vec2 and silence-based chunking.

    Loads the audio file, splits it into chunks based on silence breaks,
    transcribes each chunk using a Wav2Vec2 ASR pipeline, and logs the
    results.

    Args:
        audio_path:
            Path to the input WAV audio file.
    """
    path = Path(audio_path)

    if not path.is_file():
        logger.error("File not found: %s", audio_path)
        sys.exit(1)

    logger.info("Loading audio from %s...", audio_path)
    librosa.load(path=audio_path, sr=None)[0]

    logger.info("Creating Wav2Vec2 ASR pipeline...")
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model="CoRal-project/roest-v3-wav2vec2-315m",
        device=get_device(),
    )

    logger.info("Chunking audio...")
    for chunk in chunk_audio(audio_path=path):
        logger.info(
            "Transcribing chunk: %.2fs - %.2fs", chunk.start_time, chunk.end_time
        )
        segments = transcribe(
            audio_data=chunk.audio, pipeline=asr_pipeline, chunk_offset=chunk.start_time
        )
        for segment in segments:
            logger.info(
                "  %.2fs - %.2fs: %s",
                segment.start_time,
                segment.end_time,
                segment.text,
            )

    logger.info("Transcription complete.")


if __name__ == "__main__":
    main()
