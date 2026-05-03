"""Consolidated data models for the but_with_subs package.

This module centralises all Pydantic models, enums, and namedtuples that are shared
across multiple submodules. Each class retains its original docstring and is annotated
with a comment indicating where it was moved from.
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .constants import MAX_CHUNK_LENGTH_SECONDS


class Chunk(BaseModel):
    """A chunk of data.

    Attributes:
        start_time:
            Start time of the chunk in seconds.
        end_time:
            End time of the chunk in seconds.
        audio:
            Numpy array containing the audio data for the chunk.
        text:
            The transcribed text for this segment, or None if not available yet.
        speaker:
            The speaker name for this chunk, or None if not available.
    """

    model_config = {"arbitrary_types_allowed": True}

    start_time: float
    end_time: float
    audio: np.ndarray
    text: str | None
    speaker: str | None

    def model_post_init(self, _context: dict) -> None:
        """Post-initialisation hook for the model."""
        if self.end_time - self.start_time < MAX_CHUNK_LENGTH_SECONDS:
            raise ValueError("Duration of chunk must be at least 50ms")


class File(BaseModel):
    """Model representing downloaded media files.

    Attributes:
        url:
            The original URL that was downloaded.
        video_path:
            Path to the downloaded video file, or None if not found.
        audio_path:
            Path to the downloaded audio file, or None if not found.
    """

    url: str
    video_path: Path | None
    audio_path: Path | None


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
