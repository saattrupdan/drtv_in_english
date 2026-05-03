"""Audio chunking functionality for splitting audio into segments."""

import logging
import os
from typing import cast

import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from .constants import MIN_CHUNK_LENGTH_SECONDS
from .data_models import Chunk

logger = logging.getLogger(__package__)


def chunk_by_audio(audio: np.ndarray) -> list[Chunk]:
    """Split chunk based on the audio.

    Args:
        audio:
            Mono audio data array.

    Returns:
        List of audio chunks.
    """
    pipeline = cast(
        Pipeline,
        Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=os.getenv("HF_TOKEN")
        ),
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    pipeline.to(device)

    with ProgressHook() as hook:
        speech_timestamps = pipeline(
            dict(waveform=torch.from_numpy(audio).unsqueeze(dim=0), sample_rate=16_000),
            hook=hook,
        ).speaker_diarization

    chunks = []
    for turn, speaker in speech_timestamps:
        start_s = turn.start
        end_s = turn.end
        if end_s - start_s < MIN_CHUNK_LENGTH_SECONDS:
            continue
        chunk_audio = audio[int(start_s * 16_000) : int(end_s * 16_000)]
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
