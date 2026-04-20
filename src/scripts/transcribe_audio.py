"""Transcribe an audio file using Wav2Vec2 and generate subtitle files.

Usage:
    uv run src/scripts/transcribe_audio.py [audio_path]

The script transcribes the audio, splits it into chunks based on silence,
and outputs a `.vtt` subtitle file alongside the original audio file.
"""

import logging
import sys
from pathlib import Path

import click
from tqdm.auto import tqdm
from transformers import pipeline

from but_with_subs.chunking import chunk_audio
from but_with_subs.device import get_device
from but_with_subs.subtitling import generate_subtitles
from but_with_subs.transcribing import transcribe

logger = logging.getLogger(__package__)


MODEL_ID = "CoRal-project/roest-v3-wav2vec2-315m"


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
        logger.error("File not found: {audio_path}")
        sys.exit(1)

    print(f"Loading the {MODEL_ID} model...")
    asr_pipeline = pipeline(
        task="automatic-speech-recognition", model=MODEL_ID, device=get_device()
    )

    chunks = list(chunk_audio(audio_path=path))
    all_transcriptions = []
    for chunk in tqdm(chunks, unit="chunk", desc="Transcribing"):
        segments = transcribe(
            audio_data=chunk.audio, pipeline=asr_pipeline, chunk_offset=chunk.start_time
        )
        all_transcriptions.extend(segments)

    if not all_transcriptions:
        logger.warning("No transcription segments found. Skipping subtitle generation.")
        return

    with tqdm(total=100, unit="percent", desc="Generating subtitles") as pbar:
        for current, total in generate_subtitles(
            transcriptions=all_transcriptions, audio_path=path
        ):
            pbar.update(int(100 * current / total) - pbar.n)


if __name__ == "__main__":
    main()
