"""Audio transcription using Hugging Face automatic speech recognition.

This module provides functions to transcribe audio chunks into text segments
using a pretrained ASR pipeline from the Hugging Face transformers library.
"""

import logging

import numpy as np
from pydantic import BaseModel
from transformers import AutomaticSpeechRecognitionPipeline

logger = logging.getLogger(__package__)


class Transcription(BaseModel):
    """A transcribed text segment from an audio chunk.

    Attributes:
        start_time:
            Start time of the segment, in seconds from the beginning of the
            full audio (including any chunk offset).
        end_time:
            End time of the segment, in seconds from the beginning of the
            full audio (including any chunk offset).
        text:
            The transcribed text for this segment.
    """

    start_time: float
    end_time: float
    text: str


def transcribe(
    audio_data: np.ndarray,
    pipeline: AutomaticSpeechRecognitionPipeline,
    chunk_offset: float = 0.0,
) -> list[Transcription]:
    """Transcribe an audio chunk using an ASR pipeline.

    Runs the provided audio array through the Hugging Face automatic speech
    recognition pipeline and returns a list of ``Transcription`` models.
    The ``chunk_offset`` parameter is added to both ``start_time`` and
    ``end_time`` so that they refer to the full audio timeline rather than
    just the individual chunk.

    Args:
        audio_data:
            Mono audio data as a numpy array (16 kHz float32).
        pipeline:
            A pretrained ``AutomaticSpeechRecognitionPipeline`` instance.
        chunk_offset:
            Time offset in seconds representing where this chunk sits within
            the full audio file. Added to each segment's start/end times.

    Returns:
        A list of ``Transcription`` models, one per text segment.
    """
    logger.info(
        "Transcribing audio chunk (%.2fs offset, %d samples)",
        chunk_offset,
        audio_data.size,
    )

    result = pipeline(audio_data, return_timestamps=True)

    segments: list[Transcription] = []
    for chunk_info in result:
        chunks = chunk_info.get("chunks", [])
        for chunk in chunks:
            timestamp = chunk["timestamp"]
            start_time = float(timestamp[0]) + chunk_offset
            end_time = float(timestamp[1]) + chunk_offset
            segments.append(
                Transcription(
                    start_time=start_time,
                    end_time=end_time,
                    text=chunk["text"],
                )
            )

    logger.info("Transcribed %d segments", len(segments))
    return segments
