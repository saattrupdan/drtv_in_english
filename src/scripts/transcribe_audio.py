"""Transcribe an audio file using Wav2Vec2 and generate subtitle files.

Usage:
    uv run src/scripts/transcribe_audio.py [audio_path]

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

from but_with_subs import configure_logging
from but_with_subs.audio_chunking import chunk_by_audio
from but_with_subs.audio_loading import load_audio
from but_with_subs.constants import ASR_MODEL_ID, MAX_DURATION, MAX_WORDS
from but_with_subs.data_models import Chunk
from but_with_subs.device import get_device
from but_with_subs.subtitling import generate_subtitles
from but_with_subs.text_chunking import group_word_chunks
from but_with_subs.transcribing import transcribe_chunks_dynamic
from but_with_subs.translation import translate_chunks

logger = logging.getLogger("but_with_subs")

warnings.filterwarnings("ignore", category=UserWarning)

configure_logging()


@click.command()
@click.argument("audio_path", type=str)
@click.option(
    "--language",
    type=str,
    default="en",
    show_default=True,
    help="Target language for translation (e.g. 'en' for English).",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    show_default=True,
    help=(
        "Maximum number of chunks per batch. "
        "Higher values increase throughput but require more GPU memory."
    ),
)
@click.option(
    "--max-duration",
    type=float,
    default=MAX_DURATION,
    show_default=True,
    help=(
        "Maximum total audio duration (seconds) per batch. "
        "Lower values reduce padding waste for varied-length chunks "
        "but increase batch count."
    ),
)
def main(audio_path: str, language: str, batch_size: int, max_duration: float) -> None:
    """Transcribe an audio file using Wav2Vec2 and silence-based chunking.

    Loads the audio file, splits it into chunks based on silence breaks,
    transcribes each chunk using a Wav2Vec2 ASR pipeline with dynamic batching,
    translates the transcribed text, and outputs subtitle files.

    The dynamic batching strategy groups chunks intelligently to minimise
    padding waste while maintaining high throughput through batch processing.

    Args:
        audio_path:
            Path to the input WAV audio file.
        language:
            Target language code for translation (e.g., 'en' for English).
        batch_size:
            Maximum number of chunks per batch.
        max_duration:
            Maximum total audio duration per batch in seconds.
    """
    path = Path(audio_path)
    if not path.is_file():
        logger.error(f"File not found: {audio_path}")
        sys.exit(1)

    logger.info(f"Loading the {ASR_MODEL_ID} model...")
    with bnb.no_terminal_output():
        model = pipeline(
            task="automatic-speech-recognition",
            model=ASR_MODEL_ID,
            device=get_device(),
            num_beams=5,
        )

    logger.info("Loading the punctuation model...")
    with bnb.no_terminal_output():
        punctuation_model = PunctFixer(language="da")

    logger.info("Loading the audio file...")
    audio = load_audio(path=path)

    # Generate all chunks first
    all_chunks = chunk_by_audio(audio=audio)
    logger.info(f"Generated {len(all_chunks)} initial audio chunks")

    # Process chunks using dynamic batching with configurable parameters
    batch_results = transcribe_chunks_dynamic(
        chunks=all_chunks,
        model=model,
        batch_size=batch_size,
        max_duration=max_duration,
        show_progress=True,
    )

    # Post-process each chunk's transcription
    chunks: list[Chunk] = list()
    for word_chunks in tqdm(
        batch_results, unit="batch", desc="Processing transcriptions"
    ):
        chunked_transcriptions = group_word_chunks(
            word_chunks=word_chunks,
            punctuation_model=punctuation_model,
            max_words=MAX_WORDS,
        )
        chunks.extend(chunked_transcriptions)

    # Translate all chunks to target language
    logger.info(f"Translating transcriptions to {language}")

    translated_chunks: list[Chunk] = []
    with tqdm(total=len(chunks), desc="Translating", unit="chunk") as pbar:
        for result in translate_chunks(chunks, language, batch_size=batch_size):
            if isinstance(result, tuple):
                current, total = result
                pbar.set_description(f"Translating {current}/{total}")
                pbar.n = current
                pbar.refresh()
            else:
                translated_chunks = result
                pbar.n = pbar.total
                pbar.refresh()
        pbar.close()

    generate_subtitles(chunks=translated_chunks, audio_path=path)


if __name__ == "__main__":
    main()
