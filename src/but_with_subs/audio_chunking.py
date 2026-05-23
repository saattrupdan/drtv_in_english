"""Speaker diarisation via pyannote.audio Pipeline."""

import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from .constants import MIN_CHUNK_LENGTH_SECONDS, TARGET_SAMPLE_RATE
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


def diarize(audio: np.ndarray, model: Pipeline) -> list[tuple[float, float, str]]:
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
