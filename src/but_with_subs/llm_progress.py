"""Progress event types and shared progress utilities for LLM API calls."""

from __future__ import annotations

import collections.abc as c
import threading
import types
from typing import Self

from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm


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


class SharedProgress:
    """A thread-safe wrapper around a single ``tqdm`` progress bar.

    Designed for use across multiple parallel workers so that only one
    progress bar is visible in the terminal.
    """

    def __init__(self, total: int, desc: str = "") -> None:
        """Initialise the shared progress bar.

        Args:
            total:
                The total number of units to track.
            desc (optional):
                A description string shown alongside the progress bar.
                Defaults to an empty string.
        """
        self._lock = threading.Lock()
        self._tqdm = tqdm(total=total, desc=desc or None)

    def update(self, n: int = 1) -> None:
        """Advance the progress bar by ``n`` units.

        Args:
            n (optional):
                Number of units to advance. Defaults to 1.
        """
        with self._lock:
            self._tqdm.update(n)

    def set_description(self, desc: str) -> None:
        """Update the description shown alongside the progress bar.

        Args:
            desc:
                The new description string.
        """
        with self._lock:
            self._tqdm.set_description(desc)

    def close(self) -> None:
        """Close and clean up the progress bar."""
        with self._lock:
            self._tqdm.close()

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns:
            The ``SharedProgress`` instance itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the context manager, closing the progress bar."""
        self.close()


def SharedProgressCallback(
    shared_progress: SharedProgress,
) -> c.Callable[[LLMProgress], None]:
    """Return a callback suitable for LLM progress events.

    The callback increments the shared progress bar on completion and
    updates its description on intermediate statuses.

    Args:
        shared_progress:
            The ``SharedProgress`` instance to update.

    Returns:
        A callback function matching the ``c.Callable[[LLMProgress], None]``
        signature expected by ``query_llm`` and related functions.
    """

    def callback(progress: LLMProgress) -> None:
        if progress.status == "complete":
            shared_progress.update(1)
        elif progress.status in ("request_starting", "request_sent", "error"):
            shared_progress.set_description(progress.message)

    return callback


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
