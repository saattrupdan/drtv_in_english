"""Shared Pydantic models for the danglish package."""

from pydantic import BaseModel


class Chunk(BaseModel):
    """A single subtitle cue."""

    start_time: float
    end_time: float
    text: str | None
    speaker: str | None = None


class PrepareResponse(BaseModel):
    """Result of ``POST /prepare``: handles the browser uses to play and watch.

    Attributes:
        job_id:
            Opaque identifier for the prepared job; embedded in the
            other URLs below and used by ``GET /translate/{job_id}``.
        title:
            Episode title from DR's metadata.
        hls_url:
            Proxy URL of the HLS master playlist; play with hls.js (or
            natively in Safari).
        original_subs_url:
            Proxy URL of DR's Danish ``.vtt``; attach as the initial
            ``<track>`` while translation runs.
        cue_count:
            Number of cues in the source subtitles; used to render a
            translation progress indicator.
    """

    job_id: str
    title: str
    hls_url: str
    original_subs_url: str
    cue_count: int


class CueEvent(BaseModel):
    """One translated cue, streamed by ``GET /translate/{job_id}``.

    A final event has ``done=True`` and no cue payload.

    Attributes:
        index:
            0-based position of the cue in the source subtitles.
        start:
            Start time in seconds.
        end:
            End time in seconds.
        text:
            Translated cue text.
        done:
            True on the final sentinel event after all cues have been
            sent.
    """

    index: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    done: bool = False
