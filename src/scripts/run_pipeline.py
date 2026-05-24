"""End-to-end pipeline: URL → subtitles.

Downloads a video from a URL, extracts its audio, transcribes the audio,
corrects and translates the transcript via an LLM, writes ``.vtt`` subtitle
files next to the video, and deletes the intermediate ``.wav`` audio file.

Usage:
    uv run src/scripts/run_pipeline.py [URL] --language en
"""

import logging
import os
import sys
import warnings

import bits_and_bobs as bnb
import click
from dotenv import load_dotenv
from punctfix.inference import PunctFixer
from tqdm.auto import tqdm
from transformers import pipeline

from but_with_subs import configure_logging, download, load_diarization_pipeline
from but_with_subs.audio_extraction import extract_audio
from but_with_subs.audio_loading import load_audio
from but_with_subs.constants import ASR_MODEL_ID, MAX_WORDS
from but_with_subs.data_models import Chunk
from but_with_subs.device import get_device
from but_with_subs.llm import build_client, correct_and_translate
from but_with_subs.subtitling import generate_subtitles
from but_with_subs.text_chunking import group_word_chunks
from but_with_subs.transcribing import assign_speakers, transcribe_audio

load_dotenv()

logger = logging.getLogger("but_with_subs")

warnings.filterwarnings("ignore", category=UserWarning)

configure_logging()


@click.command()
@click.argument("url", required=True)
@click.option(
    "--language",
    type=str,
    required=True,
    help="Target language for translation (e.g. 'en' for English).",
)
def main(url: str, language: str) -> None:
    """Download a video and produce translated subtitles end-to-end."""
    logger.info(f"Downloading from {url}...")
    with tqdm(total=100, unit="%", desc="Download progress") as pbar:
        file = download(
            url=url,
            progress_hook=lambda p: pbar.update(int(100 * p.percentage - pbar.n)),
        )

    if file.video_path is None:
        logger.error("Download did not produce a video file")
        sys.exit(1)

    audio_path = extract_audio(video_path=file.video_path)

    try:
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
        audio = load_audio(path=audio_path)

        logger.info("Loading the diarisation pipeline...")
        with bnb.no_terminal_output():
            diarization_model = load_diarization_pipeline()

        word_chunks = transcribe_audio(audio=audio, model=model, show_progress=True)
        logger.info(f"Generated {len(word_chunks)} word-level segments")

        # Assign speakers via diarisation
        from but_with_subs.audio_chunking import diarize

        turns = diarize(audio, diarization_model)
        word_chunks = assign_speakers(word_chunks, turns)
        logger.info(
            f"Assigned speakers to {sum(1 for c in word_chunks if c.speaker is not None)} "
            f"of {len(word_chunks)} chunks"
        )

        chunks: list[Chunk] = group_word_chunks(
            word_chunks=word_chunks,
            punctuation_model=punctuation_model,
            max_words=MAX_WORDS,
        )
        logger.info(f"Grouped into {len(chunks)} text segments")

        # Build LLM client for correct-and-translate.
        llm_client = build_client()
        llm_model = os.environ["LLM_MODEL"]

        with tqdm(total=len(chunks), desc="Translating", unit="chunk") as pbar:

            def _on_progress(ratio: float) -> None:
                pbar.n = int(ratio * len(chunks))
                pbar.refresh()

            chunks = correct_and_translate(
                chunks,
                target_language=language,
                client=llm_client,
                model=llm_model,
                on_progress=_on_progress,
            )

        output_path = audio_path.with_suffix(".vtt")
        generate_subtitles(
            chunks=chunks, audio_path=audio_path, output_path=output_path
        )
    finally:
        if audio_path.exists():
            logger.info(f"Removing intermediate audio file: {audio_path}")
            audio_path.unlink()


if __name__ == "__main__":
    main()
