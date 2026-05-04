"""Audio transcription using Hugging Face automatic speech recognition.

This module provides functions to transcribe audio into word-level text segments
using a pretrained ASR pipeline from the Hugging Face transformers library.
"""

import typing as t

import bits_and_bobs as bnb
import numpy as np
from tqdm.auto import tqdm
from transformers import AutomaticSpeechRecognitionPipeline

from .constants import MIN_CHUNK_LENGTH_SECONDS, TARGET_SAMPLE_RATE
from .data_models import Chunk
from .logging_config import logger


def transcribe_audio(
    audio: np.ndarray,
    model: AutomaticSpeechRecognitionPipeline,
    min_chunk_length: float = MIN_CHUNK_LENGTH_SECONDS,
    show_progress: bool = True,
) -> list[Chunk]:
    """Transcribe full audio using word-level timestamps from the ASR pipeline.

    Calls the ASR pipeline once on the complete audio with ``return_timestamps="word"``
    to obtain word-level segments.  Extracts the corresponding audio slice for each
    word and filters out segments shorter than *min_chunk_length*.

    Args:
        audio:
            Full audio array (mono, float, 16 kHz).
        model:
            The ASR pipeline to use for transcription.
        min_chunk_length (optional):
            Minimum segment duration in seconds. Segments shorter than this
            are excluded. Defaults to ``MIN_CHUNK_LENGTH_SECONDS`` (0.05 s).
        show_progress (optional):
            Whether to display a progress bar. Defaults to ``True``.

    Returns:
        List of word-level transcribed ``Chunk`` objects.
    """
    with tqdm(total=1, desc="Transcribing", disable=not show_progress) as pbar:
        try:
            with bnb.no_terminal_output():
                result = t.cast(dict, model(audio, return_timestamps="word"))
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

        pbar.update(1)

    word_chunks: list[Chunk] = list()

    for transcription_dct in result["chunks"]:
        start_time = float(transcription_dct["timestamp"][0])
        end_time = float(transcription_dct["timestamp"][1])

        if end_time - start_time < min_chunk_length:
            continue

        # Extract audio for this word segment
        audio_start = int(TARGET_SAMPLE_RATE * start_time)
        audio_end = int(TARGET_SAMPLE_RATE * end_time)
        audio_end = min(audio_end, len(audio))
        segment_audio = audio[audio_start:audio_end]

        word_chunks.append(
            Chunk(
                start_time=start_time,
                end_time=end_time,
                audio=segment_audio,
                text=transcription_dct["text"],
                speaker=None,
            )
        )

    logger.info(f"Completed transcription of {len(word_chunks)} word segments")

    return word_chunks
