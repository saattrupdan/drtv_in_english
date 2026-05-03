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

from but_with_subs.audio_chunking import chunk_by_audio
from but_with_subs.audio_loading import load_audio
from but_with_subs.data_models import Chunk
from but_with_subs.device import get_device
from but_with_subs.subtitling import generate_subtitles
from but_with_subs.text_chunking import group_word_chunks
from but_with_subs.transcribing import transcribe_chunk, transcribe_chunks_batch

logger = logging.getLogger(__package__)

warnings.filterwarnings("ignore", category=UserWarning)


MODEL_ID = (
    "CoRal-project/roest-v3-whisper-1.5b"  # "CoRal-project/roest-v3-wav2vec2-315m"
)


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
            task="automatic-speech-recognition",
            model=MODEL_ID,
            device=get_device(),
            num_beams=1,
        )

    logger.info("Loading the punctuation model...")
    with bnb.no_terminal_output():
        punctuation_model = PunctFixer(language="da")

    logger.info("Loading the audio file...")
    audio = load_audio(path=path)

    # Generate all chunks first
    all_chunks = list(chunk_by_audio(audio=audio))
    logger.info("Generated %d initial audio chunks", len(all_chunks))

    # Process chunks in batches using batch transcription
    chunks: list[Chunk] = list()
    batch_results = transcribe_chunks_batch(chunks=all_chunks, model=model)

    # Post-process each chunk's transcription
    for original_chunk, word_chunks in tqdm(
        batch_results.items(),
        unit="batch",
        desc="Processing transcriptions",
    ):
        chunked_transcriptions = group_word_chunks(
            word_chunks=word_chunks,
            punctuation_model=punctuation_model,
            max_words=12,
        )
        chunks.extend(chunked_transcriptions)

    generate_subtitles(chunks=chunks, audio_path=path)


if __name__ == "__main__":
    main()
