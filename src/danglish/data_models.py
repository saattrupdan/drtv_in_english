"""Shared Pydantic models for the danglish package."""

from pathlib import Path

from pydantic import BaseModel


class Chunk(BaseModel):
    """A single subtitle cue."""

    start_time: float
    end_time: float
    text: str | None
    speaker: str | None = None


class File(BaseModel):
    """A downloaded video and its source-language subtitles."""

    url: str
    video_path: Path | None
    subtitles_path: Path | None = None


class VideoWithSubs(BaseModel):
    """Final output of the processing pipeline."""

    video_path: str
    subtitles_path: str


class ProgressEvent(BaseModel):
    """Streaming progress update emitted by the processing pipeline.

    Attributes:
        stage:
            Coarse pipeline stage (``downloading``, ``translating``,
            ``completed``, ``error``).
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
    """Progress update from yt-dlp.

    Attributes:
        status:
            Human-readable status string (e.g., downloading, finished).
        current_file:
            Name of the file currently being downloaded, or None.
        percentage:
            Download progress as a float from 0.0 to 1.0.
    """

    status: str
    current_file: str | None = None
    percentage: float
