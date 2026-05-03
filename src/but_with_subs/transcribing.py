"""Audio transcription using Hugging Face automatic speech recognition.

This module provides functions to transcribe audio chunks into text segments
using a pretrained ASR pipeline from the Hugging Face transformers library.
"""

import logging
import typing as t

import bits_and_bobs as bnb
import numpy as np
from transformers import AutomaticSpeechRecognitionPipeline

from .constants import MIN_CHUNK_LENGTH_SECONDS
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
        if end_time - start_time < MIN_CHUNK_LENGTH_SECONDS:
            continue
        audio = chunk.audio[int(16_000 * start_time) : int(16_000 * end_time)]
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


def transcribe_chunks_batch(
    chunks: list[Chunk], model: AutomaticSpeechRecognitionPipeline
) -> dict[Chunk, list[Chunk]]:
    """Transcribe multiple audio chunks in a single batch.

    This function processes multiple chunks simultaneously by:
    1. Padding shorter chunks to match the longest chunk's length
    2. Passing all chunks through the ASR pipeline in one call
    3. Mapping results back to their original input chunks

    Args:
        chunks:
            List of audio chunks to transcribe. Each chunk should have
            consistent sampling rate (16kHz expected).
        model:
            The ASR pipeline to use for transcription.

    Returns:
        A dictionary mapping each input chunk to its list of word-level
        transcribed chunks. Empty lists indicate no valid transcriptions
        were produced for that chunk.

    Notes:
        - Audio chunks of different lengths are padded with zeros to the
          length of the longest chunk.
        - Timestamps in the results are adjusted to account for the original
          chunk's start time.
        - Chunks shorter than MIN_CHUNK_LENGTH_SECONDS are skipped.
        - The batch processing is more efficient than processing chunks
          individually, especially for large numbers of chunks.
    """
    if not chunks:
        return {}

    # Find the maximum length for padding
    max_length = max(len(chunk.audio) for chunk in chunks)

    # Pad all chunks to the same length
    padded_audio_list = []
    for chunk in chunks:
        audio_length = len(chunk.audio)
        if audio_length < max_length:
            padding = np.zeros(max_length - audio_length, dtype=chunk.audio.dtype)
            padded_audio = np.concatenate([chunk.audio, padding])
        else:
            padded_audio = chunk.audio
        padded_audio_list.append(padded_audio)

    # Run batch inference
    with bnb.no_terminal_output(disable=True):
        results = t.cast(list[dict], model(padded_audio_list, return_timestamps="word"))

    # Map results back to original chunks
    chunk_transcriptions: dict[Chunk, list[Chunk]] = {}

    for chunk, result in zip(chunks, results):
        word_chunks: list[Chunk] = []
        for transcription_dct in result["chunks"]:
            start_time = float(transcription_dct["timestamp"][0]) + chunk.start_time
            end_time = float(transcription_dct["timestamp"][1]) + chunk.start_time

            if end_time - start_time < MIN_CHUNK_LENGTH_SECONDS:
                continue

            # Extract audio for this word segment
            audio_start = int(16_000 * start_time)
            audio_end = int(16_000 * end_time)
            # Ensure we don't exceed the original chunk bounds
            audio_end = min(audio_end, len(chunk.audio))
            audio = chunk.audio[audio_start:audio_end]

            word_chunks.append(
                Chunk(
                    start_time=start_time,
                    end_time=end_time,
                    audio=audio,
                    text=transcription_dct["text"],
                    speaker=chunk.speaker,
                )
            )

        chunk_transcriptions[chunk] = word_chunks

    return chunk_transcriptions
