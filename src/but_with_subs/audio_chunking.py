"""Audio chunking functionality for splitting audio into segments.

This module provides speaker diarisation via pyannote.audio Pipeline, plus a
legacy convenience function ``chunk_by_audio`` that returns audio-only segments.
"""

import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from .constants import MIN_CHUNK_LENGTH_SECONDS, TARGET_SAMPLE_RATE
from .data_models import Chunk
from .device import get_device
from .logging_config import logger


def load_diarization_pipeline() -> Pipeline:
    """Load and return a pre-warmed speaker-diarisation Pipeline.

    Returns:
        A fully-loaded ``Pipeline`` ready for ``diarize()`` calls.
    """
    model = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    assert model is not None
    model.to(get_device())
    return model


def diarize(
    audio: np.ndarray,
    model: Pipeline,
) -> list[tuple[float, float, str]]:
    """Run speaker diarisation on the given audio array.

    Args:
        audio: Mono audio array at ``TARGET_SAMPLE_RATE``.
        model: A pre-loaded diarisation Pipeline.

    Returns:
        List of ``(start, end, speaker)`` tuples for each detected turn.
    """
    with ProgressHook() as hook:
        speech_timestamps = model(
            dict(
                waveform=torch.from_numpy(audio).unsqueeze(dim=0),
                sample_rate=TARGET_SAMPLE_RATE,
            ),
            hook=hook,
        ).speaker_diarization

    turns: list[tuple[float, float, str]] = []
    for turn, speaker in speech_timestamps:
        start_s = turn.start
        end_s = turn.end
        if end_s - start_s < MIN_CHUNK_LENGTH_SECONDS:
            continue
        turns.append((start_s, end_s, speaker))

    logger.info(f"Diarisation found {len(turns)} speaker turns.")
    return turns


def chunk_by_audio(audio: np.ndarray) -> list[Chunk]:
    """Split audio into chunks based on speech timestamps.

    .. deprecated::
        Prefer :func:`load_diarization_pipeline` + :func:`diarize` + :func:`assign_speakers`.
        This function is kept as a thin convenience wrapper.

    Args:
        audio:
            Mono audio data array.

    Returns:
        List of audio chunks.
    """
    model = load_diarization_pipeline()
    turns = diarize(audio, model)

    chunks = []
    for start_s, end_s, speaker in turns:
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
