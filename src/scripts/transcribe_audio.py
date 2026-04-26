"""Transcribe an audio file using Wav2Vec2 and generate subtitle files.

Usage:
    uv run src/scripts/transcribe_audio.py [audio_path] [--language LANG]

The script transcribes the audio, splits it into chunks based on silence,
and outputs a `.vtt` subtitle file alongside the original audio file.
"""

import asyncio
import logging
import sys
from pathlib import Path

import bits_and_bobs as bnb
import click
from tqdm.auto import tqdm
from transformers import pipeline

from but_with_subs.chunking import chunk_audio
from but_with_subs.device import get_device
from but_with_subs.llm import LLMConfig
from but_with_subs.subtitling import generate_subtitles
from but_with_subs.transcribing import Transcription, transcribe
from but_with_subs.transcription_formatting import format_transcriptions
from but_with_subs.translation import translate

logger = logging.getLogger(__package__)


MODEL_ID = "CoRal-project/roest-v3-wav2vec2-315m"
# MODEL_ID = "CoRal-project/roest-v3-whisper-1.5b"


@click.command()
@click.argument("audio_path", type=str)
@click.option(
    "--llm-model",
    type=str,
    required=True,
    default="Qwen/Qwen3.6-35B-A3B-FP8",
    help="LLM model to use for translation.",
)
@click.option(
    "--llm-api-base",
    type=str,
    default="http://100.102.237.34:8000/v1",
    help="Base URL for the LLM API.",
)
@click.option("--llm-api-key", type=str, default=None, help="API key for the LLM.")
@click.option(
    "--language",
    type=str,
    default=None,
    help="Target language for translation (e.g. 'French', 'Spanish').",
)
def main(
    audio_path: str,
    language: str | None,
    llm_model: str,
    llm_api_base: str,
    llm_api_key: str | None,
) -> None:
    """Transcribe an audio file using Wav2Vec2 and silence-based chunking.

    Loads the audio file, splits it into chunks based on silence breaks,
    transcribes each chunk using a Wav2Vec2 ASR pipeline, optionally
    translates the results, and outputs subtitle files.

    Args:
        audio_path:
            Path to the input WAV audio file.
        language:
            Target language for translation. If not provided, no translation
            is performed.
        llm_model:
            Name of the LLM model to use for translation.
        llm_api_base:
            Base URL for the LLM API.
        llm_api_key:
            API key for the LLM.
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

    # Split the audio into chunks using VAD
    chunks = chunk_audio(audio_path=path)

    # Transcribe each chunk, with word-level timestamps
    chunk_transcriptions: list[list[Transcription]] = list()
    for chunk in tqdm(chunks, unit="chunk", desc="Transcribing"):
        segments = transcribe(
            audio_data=chunk.audio, model=model, chunk_offset=chunk.start_time
        )
        chunk_transcriptions.append(segments)
    if not chunk_transcriptions:
        logger.warning("No transcription segments found. Skipping subtitle generation.")
        return

    # Format the transcriptions: fix casing, punctuation, wording, and split
    # into semantically meaningful subtitle segments with accurate timestamps.
    llm_config = LLMConfig(
        model=llm_model,
        temperature=0.0,
        max_tokens=1000,
        api_base=llm_api_base,
        api_key=llm_api_key,
    )
    formatted = asyncio.run(
        format_transcriptions(
            chunk_transcriptions=chunk_transcriptions, llm_config=llm_config
        )
    )

    # Translate each segment to the target language
    segments: list[Transcription] = formatted
    if language is not None:
        segments = asyncio.run(
            translate_transcriptions(
                transcriptions=segments, target_language=language, llm_config=llm_config
            )
        )

    # Generate subtitles
    with tqdm(total=100, unit="percent", desc="Generating subtitles") as pbar:
        for current, total in generate_subtitles(
            transcriptions=segments, audio_path=path
        ):
            pbar.update(int(100 * current / total) - pbar.n)


async def translate_transcriptions(
    transcriptions: list[Transcription], target_language: str, llm_config: LLMConfig
) -> list[Transcription]:
    """Translate each transcription segment to the target language.

    Processes segments sequentially using a tqdm progress bar.

    Args:
        transcriptions:
            List of transcription segments to translate.
        target_language:
            The language to translate into.
        llm_config:
            Configuration for the LLM.

    Returns:
        A new list of ``Transcription`` objects with translated text
        but the same start and end times as the originals.
    """
    translated: list[Transcription] = []

    with tqdm(total=len(transcriptions), unit="segment", desc="Translating") as pbar:
        for segment in transcriptions:
            translated_text = await translate(
                text=segment.text,
                target_language=target_language,
                llm_config=llm_config,
            )
            translated.append(
                Transcription(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    text=translated_text,
                )
            )
            pbar.update(1)

    return translated


if __name__ == "__main__":
    main()
