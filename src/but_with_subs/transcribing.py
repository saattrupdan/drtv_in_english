"""Audio transcription using Hugging Face automatic speech recognition.

This module provides functions to transcribe audio chunks into text segments
using a pretrained ASR pipeline from the Hugging Face transformers library.
"""

import logging
import typing as t
from collections.abc import Iterator

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
    with bnb.no_terminal_output():
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


def create_dynamic_batches(
    chunks: list[Chunk],
    batch_size: int = 20,
    max_duration: float = 60.0,
) -> Iterator[list[Chunk]]:
    """Create dynamic batches from audio chunks for efficient transcription.

    This function implements an intelligent batching strategy that minimises
    padding waste by grouping chunks of similar duration together. The algorithm:

    1. Sorts chunks by duration (ascending) to group similar-length chunks
    2. Creates batches respecting both the maximum size and duration limits
    3. Yields batches one at a time for memory efficiency and progress tracking

    This approach is particularly beneficial for:
    - Reducing padding overhead when chunks have varying durations
    - Managing GPU memory more efficiently
    - Providing better progress tracking during transcription
    - Handling edge cases like very long individual chunks

    Args:
        chunks:
            List of audio chunks to batch for transcription.
        batch_size:
            Maximum number of chunks per batch. Defaults to 20, which is
            optimal for M4 Max with 64GB unified memory. Higher values
            may improve throughput but increase memory usage.
        max_duration:
            Maximum total audio duration (in seconds) per batch. Defaults to 60
            seconds, balancing processing speed with memory constraints.
            Lower values reduce padding waste but increase batch count.

    Yields:
        Lists of chunks ready for batch transcription. Each batch respects
        both the size limit (batch_size) and duration limit (max_duration).

    Example:
        >>> chunks = [...]  # List of Chunk objects
        >>> for i, batch in enumerate(create_dynamic_batches(chunks)):
        ...     results = transcribe_chunks_batch(batch, model)
        ...     logger.info(f"Processed batch {i + 1} with {len(batch)} chunks")

    Notes:
        - Very long chunks (> max_duration) will be placed in their own batch
        - Empty chunk lists produce no batches
        - Single chunks are always valid regardless of duration
        - Duration is calculated as (end_time - start_time) for each chunk
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
) -> dict[Chunk, list[Chunk]]:
    """Transcribe audio chunks using dynamic batching with progress tracking.

    This function processes audio chunks in optimised batches, intelligently
    grouping them to minimise padding waste while maintaining high throughput.
    It combines the benefits of batch processing with smart resource management.

    The batching strategy:
    1. Sorts chunks by duration to group similar-length audio together
    2. Creates batches respecting both size and duration limits
    3. Processes each batch sequentially with optional progress display
    4. Aggregates all results into a single dictionary

    Args:
        chunks:
            List of audio chunks to transcribe. Can be any size - the function
            will automatically determine optimal batch sizes.
        model:
            The ASR pipeline to use for transcription.
        batch_size:
            Maximum number of chunks per batch. Defaults to 20 for M4 Max
            with 64GB unified memory. Adjust based on available GPU memory.
        max_duration:
            Maximum total audio duration (in seconds) per batch. Defaults to 60
            seconds. Lower values reduce padding waste for varied-length chunks.
        show_progress:
            Whether to display a progress bar during processing. Defaults to True.

    Returns:
        A dictionary mapping each input chunk to its list of word-level
        transcribed chunks. Same format as transcribe_chunks_batch.

    Example:
        >>> from but_with_subs.transcribing import transcribe_chunks_dynamic
        >>> all_results = transcribe_chunks_dynamic(
        ...     chunks=all_chunks,
        ...     model=model,
        ...     batch_size=20,
        ...     max_duration=60.0
        ... )

    Performance Notes:
        - For M4 Max with 64GB RAM: batch_size=20, max_duration=60s works well
        - Larger batch sizes increase throughput but also memory usage
        - Smaller max_duration reduces padding waste but increases batch count
        - Progress tracking helps monitor long-running transcriptions
    """
    if not chunks:
        return {}

    # Create batches using the dynamic batching strategy
    batches = list(create_dynamic_batches(chunks, batch_size, max_duration))
    total_batches = len(batches)

    logger.info(
        "Processing %d chunks in %d dynamic batches (batch_size=%d, max_duration=%.1fs)",
        len(chunks),
        total_batches,
        batch_size,
        max_duration,
    )

    # Aggregate all results
    all_transcriptions: dict[Chunk, list[Chunk]] = {}

    # Create progress iterator
    batch_iterator = tqdm(
        batches,
        total=total_batches,
        desc="Transcribing batches",
        disable=not show_progress,
    )

    for batch_idx, batch in enumerate(batch_iterator):
        # Update progress bar description with batch info
        batch_duration = sum(c.end_time - c.start_time for c in batch)
        batch_iterator.set_description(
            f"Batch {batch_idx + 1}/{total_batches} ({len(batch)} chunks, {batch_duration:.1f}s)"
        )

        # Transcribe this batch
        batch_results = transcribe_chunks_batch(chunks=batch, model=model)
        all_transcriptions.update(batch_results)

    logger.info("Completed transcription of %d chunks", len(all_transcriptions))

    return all_transcriptions
