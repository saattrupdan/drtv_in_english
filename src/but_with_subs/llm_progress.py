"""Progress event types for LLM API calls."""

import collections.abc as c

from pydantic import BaseModel, ConfigDict


class LLMProgress(BaseModel):
    """Represents a progress event from an LLM API call.

    Attributes:
        status:
            The status of the progress event (e.g. "request_starting", "complete").
        elapsed_ms:
            Elapsed time in milliseconds since the start of the call.
        message:
            A human-readable description of the current state.
        model:
            The name of the model being queried, if available.
        error:
            An error message, if the call failed.
    """

    model_config = ConfigDict(frozen=True)

    status: str
    elapsed_ms: float
    message: str
    model: str | None = None
    error: str | None = None


def _noop_callback(progress: LLMProgress) -> None:
    """A no-op callback for use when no progress reporting is desired."""
    pass


def _emit_progress(
    callback: c.Callable[[LLMProgress], None] | None,
    status: str,
    elapsed_ms: float,
    message: str,
    model: str | None = None,
    error: str | None = None,
) -> None:
    """Emit a progress event if a callback is provided.

    Args:
        callback:
            The callback function to invoke, or None.
        status:
            The status string for the progress event.
        elapsed_ms:
            Elapsed time in milliseconds since the call started.
        message:
            A human-readable description of the current state.
        model (optional):
            The name of the model being queried.
        error (optional):
            An error message, if the call failed.
    """
    if callback is not None:
        callback(
            LLMProgress(
                status=status,
                elapsed_ms=elapsed_ms,
                message=message,
                model=model,
                error=error,
            )
        )
