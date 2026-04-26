"""Progress event types for LLM API calls."""

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
