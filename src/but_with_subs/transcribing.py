"""Audio transcription using Hugging Face automatic speech recognition.

This module provides functions to transcribe audio chunks into text segments
using a pretrained ASR pipeline from the Hugging Face transformers library.
"""

import logging
import typing as t

import bits_and_bobs as bnb
from transformers import AutomaticSpeechRecognitionPipeline

from .data_models import Chunk

logger = logging.getLogger(__package__)


def transcribe_chunk(
    chunk: Chunk, model: AutomaticSpeechRecognitionPipeline
) -> list[Chunk]:
    """Transcribe an audio chunk using an ASR pipeline.

    Args:
        chunk:
            A chunk of audio data.
        model:
            The transcription model.

    Returns:
        The chunk split into words with transcriptions.
    """
    with bnb.no_terminal_output(disable=True):
        result = t.cast(dict, model(chunk.audio, return_timestamps="word"))

    word_chunks: list[Chunk] = list()
    for transcription_dct in result["chunks"]:
        start_time = float(transcription_dct["timestamp"][0]) + chunk.start_time
        end_time = float(transcription_dct["timestamp"][1]) + chunk.start_time
        audio = chunk.audio[16_000 * start_time : 16_000 * end_time]
        word_chunks.append(
            Chunk(
                start_time=start_time,
                end_time=end_time,
                audio=audio,
                text=transcription_dct["text"],
                speaker=chunk.speaker,
            )
        )

    return word_chunks
