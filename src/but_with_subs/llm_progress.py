"""LLM progress tracking for subtitling operations.

Provides ``LLMProgress`` to signal LLM operation status changes
during transcription formatting.
"""

from pydantic import BaseModel


class LLMProgress(BaseModel):
    """A progress update from an LLM operation.

    Attributes:
        status:
            One of ``"request_starting"``, ``"request_sent"``,
            ``"complete"``, or ``"error"``.
    """

    status: str
