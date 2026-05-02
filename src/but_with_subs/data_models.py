"""Consolidated data models for the but_with_subs package.

This module centralises all Pydantic models, enums, and namedtuples that are shared
across multiple submodules. Each class retains its original docstring and is annotated
with a comment indicating where it was moved from.
"""

from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel


class LLMServerType(str, Enum):
    """Detected type of LLM backend server."""

    LLAMA_CPP = "llama_cpp"
    OPENAI_COMPATIBLE = "openai_compatible"
    UNKNOWN = "unknown"


class LLMConfig(BaseModel):
    """Configuration for an LLM API call.

    Attributes:
        model:
            The name of the LLM model to use.
        temperature:
            The temperature to use for the LLM API. Required.
        max_tokens:
            The maximum number of tokens to generate.
        api_base:
            The base URL of the LLM API. Required.
        api_key:
            The API key to use for the LLM API. Not required for local LLMs.
        server_type:
            The type of LLM backend server. Auto-detected on first query
            if not explicitly set. Can be manually set to override autodetection.
        response_model:
            A Pydantic BaseModel subclass that will be used to parse the response. Can
            be None if no structured generation is used.
    """

    model: str
    temperature: float
    max_tokens: int
    api_base: str
    api_key: str | None = None
    server_type: LLMServerType | None = None
    response_model: type[BaseModel] | None = None


class QueryLLMBatchItem(NamedTuple):
    """A single item in a batch LLM query.

    Attributes:
        prompt:
            The prompt text to send to the LLM.
        config:
            Configuration for the LLM API call.
    """

    prompt: str
    config: LLMConfig


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


class Chunk(BaseModel):
    """A chunk of audio data.

    Attributes:
        start_time:
            Start time of the chunk in seconds.
        end_time:
            End time of the chunk in seconds.
        audio:
            Numpy array containing the audio data for the chunk.
        speaker:
            The speaker name for this chunk, or None if not available.
    """

    model_config = {"arbitrary_types_allowed": True}

    start_time: float
    end_time: float
    audio: np.ndarray
    speaker: str | None


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


class TranscribedSegmentsResponse(BaseModel):
    """Response containing a list of formatted segments.

    Attributes:
        segments:
            List of formatted subtitle segments.
    """

    segments: list[Transcription]


class TranslatedText(BaseModel):
    """A model representing translated text."""

    text: str = ""
