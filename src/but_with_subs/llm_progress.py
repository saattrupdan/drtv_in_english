"""LLM progress tracking dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMProgress:
    """Progress information for an LLM call.

    Attributes:
        model_name:
            The name of the LLM model being used.
        progress:
            Current progress as a float between 0.0 and 1.0.
        tokens_so_far:
            Number of tokens processed so far.
        time_elapsed:
            Time elapsed since the LLM call started, in seconds.
    """

    model_name: str
    progress: float
    tokens_so_far: int
    time_elapsed: float
