"""Audio chunking functionality for splitting audio into segments."""

import logging
import pathlib as pl
import typing as t
from collections.abc import Generator

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__package__)


class Chunk(BaseModel):
    """A chunk of audio data.

    Attributes:
        start_time:
            Start time of the chunk, in seconds from the beginning of the audio.
        end_time:
            End time of the chunk, in seconds from the beginning of the audio.
        audio:
            Mono audio data of the chunk, as a numpy array in 16 kHz.
    """

    start_time: float
    end_time: float
    audio: np.ndarray
