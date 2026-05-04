"""Audio chunking functionality for splitting audio into segments."""

import typing as t

import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from .constants import MIN_CHUNK_LENGTH_SECONDS, TARGET_SAMPLE_RATE
from .data_models import Chunk
from .device import get_device
from .logging_config import logger


def chunk_by_audio(audio: np.ndarray) -> list[Chunk]:
    """Split audio into chunks based on speech timestamps.

    Args:
        audio:
            Mono audio data array.

    Returns:
        List of audio chunks.
    """
    model = t.cast(
        Pipeline, Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    )
    model.to(get_device())

    with ProgressHook() as hook:
        speech_timestamps = model(
            dict(
                waveform=torch.from_numpy(audio).unsqueeze(dim=0),
                sample_rate=TARGET_SAMPLE_RATE,
            ),
            hook=hook,
        ).speaker_diarization

    chunks = []
    for turn, speaker in speech_timestamps:
        start_s = turn.start
        end_s = turn.end
        if end_s - start_s < MIN_CHUNK_LENGTH_SECONDS:
            continue
        chunk_audio = audio[
            int(start_s * TARGET_SAMPLE_RATE) : int(end_s * TARGET_SAMPLE_RATE)
        ]
        chunk = Chunk(
            start_time=start_s,
            end_time=end_s,
            audio=chunk_audio,
            text=None,
            speaker=speaker,
        )
        chunks.append(chunk)

    logger.info(f"Split audio into {len(chunks)} chunks.")
    return chunks
