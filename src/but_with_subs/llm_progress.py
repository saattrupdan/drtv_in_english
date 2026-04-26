"""Progress event types for LLM request lifecycle."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMProgress:
    """Progress event for LLM request lifecycle."""

    status: t.Literal["request_starting", "request_sent", "complete", "error"]
    elapsed_ms: float
    message: str
    model: str | None = None
    error: str | None = None


def _noop_callback(_: LLMProgress) -> None:
    """No-op progress callback used as default."""
    pass
