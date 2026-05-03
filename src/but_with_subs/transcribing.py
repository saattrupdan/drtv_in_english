"""Audio transcription using Hugging Face automatic speech recognition.

This module provides functions to transcribe audio chunks into text segments
using a pretrained ASR pipeline from the Hugging Face transformers library.
"""

import logging
import typing as t
from typing import Generator

import bits_and_bobs as bnb
import numpy as np
from tqdm.auto import tqdm
from transformers import AutomaticSpeechRecognitionPipeline

from .constants import MIN_CHUNK_LENGTH_SECONDS
from .data_models import Chunk

logger = logging.getLogger(__package__)


def _transcribe_chunks_batch(
    chunks: list[Chunk], model: AutomaticSpeechRecognitionPipeline
) -> list[list[Chunk]]:
    """Transcribe multiple audio chunks in a single batch.

    Returns:
        List of lists where each inner list contains word-level Chunk
        transcriptions for the corresponding input chunk. Empty lists
        indicate no valid transcriptions were produced.
    """
    if not chunks:
        return list()

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
    try:
        with bnb.no_terminal_output():
            results = t.cast(list[dict], model(padded_audio_list, return_timestamps="word"))
    except Exception as e:
        logger.error(f"Batch transcription failed: {e}")
        raise

    chunk_transcriptions: list[list[Chunk]] = list()
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

        chunk_transcriptions.append(word_chunks)

    return chunk_transcriptions


def create_dynamic_batches(
    chunks: list[Chunk], batch_size: int = 20, max_duration: float = 60.0
) -> Generator[list[Chunk], None, None]:
    """Create dynamic batches from audio chunks for efficient transcription.

    Groups chunks of similar duration to minimize padding waste during batch
    transcription. Yields batches respecting both size and duration limits.

    Args:
        chunks: List of audio chunks to batch.
        batch_size: Maximum chunks per batch. Defaults to 20.
        max_duration: Maximum total duration (seconds) per batch. Defaults to 60.

    Yields:
        Lists of chunks ready for batch transcription.
    """
    if not chunks:
        return

    # Sort chunks by duration (ascending) to group similar-length chunks together
    # This minimises padding waste within each batch
    sorted_chunks = sorted(chunks, key=lambda c: c.end_time - c.start_time)

    current_batch: list[Chunk] = []
    current_duration = 0.0

    for chunk in sorted_chunks:
        chunk_duration = chunk.end_time - chunk.start_time

        # Check if adding this chunk would exceed limits
        would_exceed_size = len(current_batch) >= batch_size
        would_exceed_duration = (
            current_duration + chunk_duration > max_duration and len(current_batch) > 0
        )

        # Yield current batch if limits would be exceeded
        if would_exceed_size or would_exceed_duration:
            yield current_batch
            current_batch = []
            current_duration = 0.0

        # Add chunk to current batch
        current_batch.append(chunk)
        current_duration += chunk_duration

    # Yield any remaining chunks
    if current_batch:
        yield current_batch


def transcribe_chunks_dynamic(
    chunks: list[Chunk],
    model: AutomaticSpeechRecognitionPipeline,
    batch_size: int = 20,
    max_duration: float = 60.0,
    show_progress: bool = True,
) -> list[list[Chunk]]:
    """Transcribe audio chunks using dynamic batching with progress tracking.

    Groups chunks by duration to minimise padding waste during batch transcription.

    Args:
        chunks: List of audio chunks to transcribe.
        model: The ASR pipeline to use for transcription.
        batch_size: Maximum chunks per batch. Defaults to 20.
        max_duration: Maximum total duration (seconds) per batch. Defaults to 60.
        show_progress: Whether to display a progress bar. Defaults to True.

    Returns:
        List of lists containing word-level transcribed chunks for each input.
    """
    if not chunks:
        return list()

    # Create batches using the dynamic batching strategy
    batches = list(create_dynamic_batches(chunks, batch_size, max_duration))
    total_batches = len(batches)

    logger.info(
        f"Processing {len(chunks)} chunks in {total_batches} dynamic batches "
        f"(batch_size={batch_size}, max_duration={max_duration:.1f}s)"
    )

    all_transcriptions: list[list[Chunk]] = list()
    with tqdm(
        batches,
        total=total_batches,
        desc="Transcribing batches",
        disable=not show_progress,
    ) as batch_iterator:
        for batch_idx, batch in enumerate(batch_iterator):
            # Update progress bar description with batch info
            batch_duration = sum(c.end_time - c.start_time for c in batch)
            batch_iterator.set_description(
                f"Batch {batch_idx + 1}/{total_batches} "
                f"({len(batch)} chunks, {batch_duration:.1f}s)"
            )

            # Transcribe this batch
            try:
                batch_results = _transcribe_chunks_batch(chunks=batch, model=model)
            except Exception as e:
                logger.error(f"Transcription failed for batch {batch_idx + 1}: {e}")
                raise
            all_transcriptions.extend(batch_results)

    logger.info(f"Completed transcription of {len(all_transcriptions)} chunks")

    return all_transcriptions
