"""Transcribe an audio file using Wav2Vec2 and generate subtitle files.

Usage:
    uv run src/scripts/transcribe_audio.py [audio_path]

The script transcribes the full audio in one pass and outputs a ``.vtt``
subtitle file alongside the original audio file.
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
from but_with_subs.audio_loading import load_audio
from but_with_subs.constants import ASR_MODEL_ID, MAX_WORDS
from but_with_subs.data_models import Chunk
from but_with_subs.device import get_device
from but_with_subs.subtitling import generate_subtitles
from but_with_subs.text_chunking import group_word_chunks
from but_with_subs.transcribing import transcribe_audio
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
def main(audio_path: str, language: str) -> None:
    """Transcribe an audio file using Wav2Vec2 and generate subtitles."""
    path = Path(audio_path)
    if not path.is_file():
        logger.error(f"File not found: {audio_path}")
        sys.exit(1)

    if not path.suffix == ".wav":
        logger.error(f"The file must be a wav file, but received {path.suffix}")
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

    # Transcribe the full audio in one pass
    word_chunks = transcribe_audio(
        audio=audio,
        model=model,
        show_progress=True,
    )
    logger.info(f"Generated {len(word_chunks)} word-level segments")

    # Group word-level chunks into readable segments
    chunks: list[Chunk] = group_word_chunks(
        word_chunks=word_chunks,
        punctuation_model=punctuation_model,
        max_words=MAX_WORDS,
    )
    logger.info(f"Grouped into {len(chunks)} text segments")

    # Translate all chunks to target language
    translated_chunks: list[Chunk] = []
    with tqdm(total=len(chunks), desc="Translating", unit="chunk") as pbar:
        for result in translate_chunks(chunks, language, batch_size=16):
            if isinstance(result, tuple):
                current, total = result
                pbar.set_description(f"Translating {current}/{total}")
                pbar.n = current
                pbar.refresh()
            else:
                translated_chunks.append(result)
                pbar.n = pbar.total
                pbar.refresh()
        pbar.close()

    output_path = path.with_suffix(".vtt")

    generate_subtitles(
        chunks=chunks, audio_path=path, output_path=output_path.with_suffix(".da.vtt")
    )
    generate_subtitles(
        chunks=translated_chunks,
        audio_path=path,
        output_path=output_path.with_suffix(f".{language}.vtt"),
    )


if __name__ == "__main__":
    main()
