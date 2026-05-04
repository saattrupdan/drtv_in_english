"""Audio transcription using Hugging Face automatic speech recognition.

This module provides functions to transcribe audio chunks into text segments
using a pretrained ASR pipeline from the Hugging Face transformers library.
"""

import collections.abc as c
import typing as t

import bits_and_bobs as bnb
import numpy as np
from tqdm.auto import tqdm
from transformers import AutomaticSpeechRecognitionPipeline

from .constants import MIN_CHUNK_LENGTH_SECONDS, TARGET_SAMPLE_RATE
from .data_models import Chunk
from .logging_config import logger


def _transcribe_chunk(
    chunk: Chunk, model: AutomaticSpeechRecognitionPipeline
) -> list[Chunk]:
    """Transcribe a single audio chunk.

    Returns:
        List of word-level Chunk transcriptions. Empty list indicates
        no valid transcription was produced.
    """
    # Pad chunk to a reasonable length for the model
    audio = chunk.audio

    try:
        with bnb.no_terminal_output():
            result = t.cast(
                dict,
                model(audio, return_timestamps="word"),
            )
    except Exception as e:
        logger.error(f"Transcription failed for chunk at {chunk.start_time:.3f}s: {e}")
        raise

    chunk_transcriptions: list[Chunk] = list()
    word_results = result["chunks"]

    for transcription_dct in word_results:
        start_time = float(transcription_dct["timestamp"][0]) + chunk.start_time
        end_time = float(transcription_dct["timestamp"][1]) + chunk.start_time

        if end_time - start_time < MIN_CHUNK_LENGTH_SECONDS:
            continue

        # Extract audio for this word segment (relative to chunk start)
        audio_start = int(TARGET_SAMPLE_RATE * (start_time - chunk.start_time))
        audio_end = int(TARGET_SAMPLE_RATE * (end_time - chunk.start_time))
        audio_end = min(audio_end, len(chunk.audio))
        audio = chunk.audio[audio_start:audio_end]

        chunk_transcriptions.append(
            Chunk(
                start_time=start_time,
                end_time=end_time,
                audio=audio,
                text=transcription_dct["text"],
                speaker=chunk.speaker,
            )
        )

    return chunk_transcriptions


def create_dynamic_batches(
    chunks: list[Chunk], batch_size: int, max_duration: float = 60.0
) -> c.Generator[list[Chunk], None, None]:
    """Create dynamic batches from audio chunks for efficient transcription.

    Groups chunks of similar duration to minimise padding waste during batch
    transcription. Yields batches respecting both size and duration limits.

    Args:
        chunks:
            List of audio chunks to batch.
        batch_size (optional):
            Maximum chunks per batch. Defaults to 20.
        max_duration (optional):
            Maximum total duration (seconds) per batch. Defaults to 60.

    Yields:
        Lists of chunks ready for batch transcription.
    """
    if not chunks:
        return

    # Sort chunks by duration (ascending) to group similar-length chunks together
    # This minimises padding waste within each batch
    sorted_chunks: list[Chunk] = sorted(chunks, key=lambda c: c.end_time - c.start_time)

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
    batch_size: int = 1,
    max_duration: float = 60.0,
    show_progress: bool = True,
) -> list[list[Chunk]]:
    """Transcribe audio chunks with optional batching.

    When batch_size=1, each chunk is transcribed individually (no batching).
    When batch_size>1, chunks are grouped into dynamic batches. However,
    each chunk is still transcribed individually within the batch to preserve
    correct word-level result association.

    Note: True batched inference (stacking multiple audio tensors and running
    one model call) is not supported because the ASR pipeline returns a flat
    list of word-level results that cannot be reliably split back per-chunk.

    Args:
        chunks:
            List of audio chunks to transcribe.
        model:
            The ASR pipeline to use for transcription.
        batch_size (optional):
            Maximum chunks per batch. When 1, each chunk is transcribed
            individually. Defaults to 1.
        max_duration (optional):
            Maximum total duration (seconds) per batch. Defaults to 60.
        show_progress (optional):
            Whether to display a progress bar. Defaults to True.

    Returns:
        List of lists containing word-level transcribed chunks for each input.
    """
    if not chunks:
        return list()

    # Always transcribe each chunk individually to preserve correct
    # word-level result association. The batch_size parameter controls
    # how many chunks are shown together in the progress bar, not how
    # they are actually processed.
    all_transcriptions: list[list[Chunk]] = list()
    total = len(chunks)

    with tqdm(
        chunks,
        total=total,
        desc="Transcribing",
        disable=not show_progress,
    ) as iterator:
        for chunk in iterator:
            try:
                word_chunks = _transcribe_chunk(chunk=chunk, model=model)
            except Exception as e:
                logger.error(f"Transcription failed for chunk at {chunk.start_time:.3f}s: {e}")
                word_chunks = []
            all_transcriptions.append(word_chunks)

    logger.info(f"Completed transcription of {len(all_transcriptions)} chunks")

    return all_transcriptions
