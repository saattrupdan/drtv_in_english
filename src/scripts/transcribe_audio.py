"""Transcribe an audio file using Wav2Vec2 and generate subtitle files.

Usage:
    uv run src/scripts/transcribe_audio.py [audio_path] [--language LANG]

The script transcribes the audio, splits it into chunks based on silence,
and outputs a `.vtt` subtitle file alongside the original audio file.
"""

import logging
import sys
import warnings
from pathlib import Path

import bits_and_bobs as bnb
import click
from punctfix.inference import PunctFixer
from tqdm.auto import tqdm
from transformers import pipeline

from but_with_subs.audio_chunking import chunk_audio
from but_with_subs.audio_loading import load_audio
from but_with_subs.data_models import Transcription
from but_with_subs.device import get_device
from but_with_subs.subtitling import generate_subtitles
from but_with_subs.text_chunking import chunk_transcriptions
from but_with_subs.transcribing import transcribe

logger = logging.getLogger(__package__)

warnings.filterwarnings("ignore", category=UserWarning)


MODEL_ID = "CoRal-project/roest-v3-wav2vec2-315m"


@click.command()
@click.argument("audio_path", type=str)
@click.option(
    "--language",
    type=str,
    default=None,
    help="Target language for translation (e.g. 'French', 'Spanish').",
)
def main(audio_path: str, language: str | None) -> None:
    """Transcribe an audio file using Wav2Vec2 and silence-based chunking.

    Loads the audio file, splits it into chunks based on silence breaks,
    transcribes each chunk using a Wav2Vec2 ASR pipeline, and outputs
    subtitle files.

    Args:
        audio_path:
            Path to the input WAV audio file.
        language (optional):
            Target language for translation. If not provided, no translation
            is performed.
    """
    path = Path(audio_path)
    if not path.is_file():
        logger.error("File not found: %s", audio_path)
        sys.exit(1)

    logger.info("Loading the %s model...", MODEL_ID)
    with bnb.no_terminal_output():
        model = pipeline(
            task="automatic-speech-recognition", model=MODEL_ID, device=get_device()
        )

    logger.info("Loading the punctuation model...")
    with bnb.no_terminal_output():
        punctuation_model = PunctFixer(language="da")

    logger.info("Loading the audio file...")
    audio = load_audio(path=path)

    transcriptions: list[Transcription] = list()
    for chunk in tqdm(chunk_audio(audio=audio), unit="chunk", desc="Transcribing"):
        word_transcriptions = transcribe(
            audio_data=chunk.audio, model=model, chunk_offset=chunk.start_time
        )
        chunked_transcriptions = chunk_transcriptions(
            transcriptions=word_transcriptions,
            punctuation_model=punctuation_model,
            max_words=12,
        )
        transcriptions.extend(chunked_transcriptions)
    if not transcriptions:
        logger.warning("No transcription segments found. Skipping subtitle generation.")
        return

    generate_subtitles(transcriptions=transcriptions, audio_path=path)


if __name__ == "__main__":
    main()
