"""Consolidated data models for the but_with_subs package.

This module centralises all Pydantic models, enums, and namedtuples that are shared
across multiple submodules. Each class retains its original docstring and is annotated
with a comment indicating where it was moved from.
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel


class Chunk(BaseModel):
    """A chunk of audio data with transcription metadata."""

    model_config = {"arbitrary_types_allowed": True}

    start_time: float
    end_time: float
    audio: np.ndarray
    text: str | None
    speaker: str | None


class File(BaseModel):
    """Model representing downloaded media files."""

    url: str
    video_path: Path | None
    audio_path: Path | None
    subtitles_path: Path | None = None


class VideoWithSubs(BaseModel):
    """Final output of the processing pipeline."""

    video_path: str
    subtitles_path: str


class ProgressEvent(BaseModel):
    """Streaming progress update emitted by the processing pipeline.

    Attributes:
        stage:
            Coarse pipeline stage (``downloading``, ``transcribing``,
            ``subtitling``, ``completed``, ``error``).
        percentage:
            Overall progress as a float in ``[0, 100]``.
        message:
            Optional human-readable status message.
        result:
            Populated only on the final ``completed`` event.
    """

    stage: str
    percentage: float
    message: str | None = None
    result: VideoWithSubs | None = None


class DownloadProgress(BaseModel):
    """Model representing download progress.

    Attributes:
        status:
            Human-readable status string (e.g., downloading, finished).
        current_file:
            Name of the file currently being downloaded, or None.
        percentage:
            Download progress as a float from 0.0 to 100.0.
    """

    status: str
    current_file: str | None = None
    percentage: float
