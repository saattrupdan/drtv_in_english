"""Tests for the speaker assignment (diarisation) logic.

These tests validate the ``assign_speakers`` helper in
``but_with_subs.transcribing``, which maps word-level chunks to
speakers based on temporal overlap with diarisation turns.
"""

import numpy as np
import pytest

from but_with_subs.data_models import Chunk
from but_with_subs.transcribing import assign_speakers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_chunks() -> list[Chunk]:
    """Create a list of mock word-level chunks spanning 0–5 seconds."""
    return [
        Chunk(
            start_time=0.0,
            end_time=1.0,
            audio=np.zeros(16000, dtype=np.float32),
            text="Hej",
            speaker=None,
        ),
        Chunk(
            start_time=1.0,
            end_time=2.0,
            audio=np.zeros(16000, dtype=np.float32),
            text="verden",
            speaker=None,
        ),
        Chunk(
            start_time=2.0,
            end_time=3.0,
            audio=np.zeros(16000, dtype=np.float32),
            text="hvad",
            speaker=None,
        ),
        Chunk(
            start_time=3.0,
            end_time=4.0,
            audio=np.zeros(16000, dtype=np.float32),
            text="så",
            speaker=None,
        ),
        Chunk(
            start_time=4.0,
            end_time=5.0,
            audio=np.zeros(16000, dtype=np.float32),
            text="jeg",
            speaker=None,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_assign_speakers_single_speaker(mock_chunks: list[Chunk]) -> None:
    """When there is one speaker turn covering the whole audio, all chunks get that speaker."""
    turns: list[tuple[float, float, str]] = [(0.0, 5.0, "Alice")]
    result = assign_speakers(mock_chunks, turns)

    for chunk in result:
        assert chunk.speaker == "Alice"


def test_assign_speakers_multiple_turns(mock_chunks: list[Chunk]) -> None:
    """When turns cover different segments, speakers are assigned correctly."""
    turns: list[tuple[float, float, str]] = [
        (0.0, 2.0, "Alice"),
        (2.0, 4.0, "Bob"),
        (4.0, 5.0, "Alice"),
    ]
    result = assign_speakers(mock_chunks, turns)

    assert result[0].speaker == "Alice"  # 0–1
    assert result[1].speaker == "Alice"  # 1–2 (overlaps Alice 0-2)
    assert result[2].speaker == "Bob"    # 2–3 (overlaps Bob 2-4)
    assert result[3].speaker == "Bob"    # 3–4 (overlaps Bob 2-4)
    assert result[4].speaker == "Alice"  # 4–5 (overlaps Alice 4-5)


def test_assign_speakers_overlapping_turns(mock_chunks: list[Chunk]) -> None:
    """When turns overlap, the chunk gets the speaker with the largest overlap."""
    turns: list[tuple[float, float, str]] = [
        (0.0, 3.0, "Alice"),
        (1.0, 4.0, "Bob"),
    ]
    result = assign_speakers(mock_chunks, turns)

    # Chunk 0 (0–1): overlaps Alice 0-1 (1s) vs Bob 1-1 (0s) → Alice
    assert result[0].speaker == "Alice"
    # Chunk 1 (1–2): overlaps Alice 1-2 (1s) vs Bob 1-2 (1s) → tie → first found wins
    # But our implementation picks the one with largest overlap; they're equal,
    # so whichever comes first in the list gets assigned.
    # Chunk 2 (2–3): overlaps Alice 2-3 (1s) vs Bob 2-3 (1s) → tie → whichever first
    # Chunk 3 (3–4): overlaps Alice 3-3 (0s) vs Bob 3-4 (1s) → Bob
    assert result[3].speaker == "Bob"


def test_assign_speakers_empty_turns(mock_chunks: list[Chunk]) -> None:
    """When there are no turns, speakers remain None."""
    turns: list[tuple[float, float, str]] = []
    result = assign_speakers(mock_chunks, turns)

    for chunk in result:
        assert chunk.speaker is None


def test_assign_speakers_empty_chunks() -> None:
    """When there are no chunks, returns an empty list."""
    result = assign_speakers([], [(0.0, 5.0, "Alice")])
    assert result == []


def test_assign_speakers_chunk_partial_overlap(mock_chunks: list[Chunk]) -> None:
    """A chunk that only partially overlaps a turn should get that speaker."""
    turns: list[tuple[float, float, str]] = [
        (0.5, 2.5, "Alice"),
    ]
    result = assign_speakers(mock_chunks, turns)

    # Chunk 0 (0–1): overlaps Alice 0.5-1 (0.5s) → Alice
    assert result[0].speaker == "Alice"
    # Chunk 1 (1–2): overlaps Alice 1-2 (1s) → Alice
    assert result[1].speaker == "Alice"
    # Chunk 2 (2–3): overlaps Alice 2-2.5 (0.5s) → Alice
    assert result[2].speaker == "Alice"
    # Chunk 3 (3–4): no overlap → None
    assert result[3].speaker is None


def test_assign_speakers_preserves_chunk_identity(mock_chunks: list[Chunk]) -> None:
    """assign_speakers should return the same Chunk objects (mutated in-place)."""
    turns: list[tuple[float, float, str]] = [(0.0, 5.0, "Alice")]
    result = assign_speakers(mock_chunks, turns)

    for i, chunk in enumerate(result):
        assert chunk is mock_chunks[i]


def test_assign_speakers_handles_single_word_chunk(mock_chunks: list[Chunk]) -> None:
    """A single small chunk should be assigned correctly."""
    single_chunk = [Chunk(
        start_time=2.0,
        end_time=2.5,
        audio=np.zeros(8000, dtype=np.float32),
        text="word",
        speaker=None,
    )]
    turns: list[tuple[float, float, str]] = [(1.0, 3.0, "Bob")]
    result = assign_speakers(single_chunk, turns)

    assert result[0].speaker == "Bob"
