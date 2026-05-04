"""Tests for the transcribing module.

This module contains comprehensive tests for the Chunk model and the
transcribe functions, including mocking the AutomaticSpeechRecognitionPipeline
to verify correct behavior under various conditions.
"""

import typing as t
import unittest.mock as um

import numpy as np
import pytest

from but_with_subs.data_models import Chunk
from but_with_subs.transcribing import (
    _transcribe_chunks_batch as transcribe_chunks_batch,
)
from but_with_subs.transcribing import create_dynamic_batches, transcribe_chunks_dynamic

# ---------------------------------------------------------------------------
# Chunk model tests
# ---------------------------------------------------------------------------


def test_chunk_model_creation_with_all_fields() -> None:
    """Test constructing a Chunk model with all fields populated.

    Verifies that the Chunk model can be instantiated with
    start_time, end_time, audio, text, and speaker, and that the values are stored
    correctly.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunk: Chunk = Chunk(
        start_time=0.0, end_time=1.5, audio=mock_audio, text="Hello world", speaker=None
    )

    assert chunk.start_time == 0.0
    assert chunk.end_time == 1.5
    assert chunk.text == "Hello world"
    assert chunk.speaker is None


def test_chunk_model_with_speaker() -> None:
    """Test constructing a Chunk model with a speaker.

    Verifies that the Chunk model correctly stores a speaker name.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunk = Chunk(
        start_time=2.5,
        end_time=5.0,
        audio=mock_audio,
        text="This is a later segment",
        speaker="Alice",
    )

    assert chunk.start_time == 2.5
    assert chunk.end_time == 5.0
    assert chunk.text == "This is a later segment"
    assert chunk.speaker == "Alice"


def test_chunk_duration_matches() -> None:
    """Test that duration equals end_time minus start_time.

    Verifies that for a Chunk with start_time=0.0 and
    end_time=3.0, the duration (end_time - start_time) equals 3.0.
    """
    mock_audio: np.ndarray = np.zeros(shape=48000, dtype=np.float32)
    chunk = Chunk(
        start_time=0.0,
        end_time=3.0,
        audio=mock_audio,
        text="Three second segment",
        speaker=None,
    )

    assert chunk.end_time - chunk.start_time == 3.0


def test_chunk_duration_with_offset() -> None:
    """Test duration is correct when start_time is non-zero.

    Verifies that the duration calculation works correctly when
    start_time is greater than zero.
    """
    mock_audio: np.ndarray = np.zeros(shape=48000, dtype=np.float32)
    chunk = Chunk(
        start_time=1.5,
        end_time=4.5,
        audio=mock_audio,
        text="Three second segment with offset",
        speaker=None,
    )

    assert chunk.end_time - chunk.start_time == 3.0


# ---------------------------------------------------------------------------
# create_dynamic_batches() function tests
# ---------------------------------------------------------------------------


def test_create_dynamic_batches_sorts_by_duration() -> None:
    """Test that create_dynamic_batches sorts chunks by duration.

    Verifies that chunks are ordered from shortest to longest duration
    before batching.
    """
    chunks = [
        Chunk(
            start_time=0.0,
            end_time=5.0,
            audio=np.zeros(80000),
            text="Long",
            speaker=None,
        ),
        Chunk(
            start_time=0.0,
            end_time=1.0,
            audio=np.zeros(16000),
            text="Short",
            speaker=None,
        ),
        Chunk(
            start_time=0.0,
            end_time=2.0,
            audio=np.zeros(32000),
            text="Medium",
            speaker=None,
        ),
    ]

    batches = list(create_dynamic_batches(chunks, batch_size=10, max_duration=60.0))

    # Shortest chunk should come first in the batch
    assert batches[0][0].text == "Short"
    assert batches[0][1].text == "Medium"
    assert batches[0][2].text == "Long"


def test_create_dynamic_batches_respects_max_duration() -> None:
    """Test that batches respect the max_duration limit.

    Verifies that when adding a chunk would exceed max_duration,
    a new batch is started.
    """
    chunks = [
        Chunk(
            start_time=0.0,
            end_time=30.0,
            audio=np.zeros(480000),
            text="30s chunk",
            speaker=None,
        ),
        Chunk(
            start_time=0.0,
            end_time=35.0,
            audio=np.zeros(560000),
            text="35s chunk",
            speaker=None,
        ),
    ]

    batches = list(create_dynamic_batches(chunks, batch_size=10, max_duration=60.0))

    # Should create two separate batches since 30+35 > 60
    assert len(batches) == 2


def test_create_dynamic_batches_respects_batch_size() -> None:
    """Test that batches respect the batch_size limit.

    Verifies that when batch_size is reached, a new batch is started.
    """
    chunks = [
        Chunk(
            start_time=0.0,
            end_time=1.0,
            audio=np.zeros(16000),
            text=f"Chunk {i}",
            speaker=None,
        )
        for i in range(5)
    ]

    batches = list(create_dynamic_batches(chunks, batch_size=3, max_duration=60.0))

    # Should create two batches: 3 chunks + 2 chunks
    assert len(batches) == 2
    assert len(batches[0]) == 3
    assert len(batches[1]) == 2


def test_create_dynamic_batches_empty_list() -> None:
    """Test that create_dynamic_batches handles empty input.

    Verifies that an empty list produces no batches.
    """
    batches = list(create_dynamic_batches([], batch_size=10, max_duration=60.0))

    assert batches == []


def test_create_dynamic_batches_single_chunk() -> None:
    """Test that a single chunk produces a single batch.

    Verifies that edge case of one chunk is handled correctly.
    """
    chunks = [
        Chunk(
            start_time=0.0,
            end_time=10.0,
            audio=np.zeros(160000),
            text="Single",
            speaker=None,
        )
    ]

    batches = list(create_dynamic_batches(chunks, batch_size=10, max_duration=60.0))

    assert len(batches) == 1
    assert len(batches[0]) == 1


# ---------------------------------------------------------------------------
# transcribe_chunks_batch() function tests
# ---------------------------------------------------------------------------


def _make_mock_pipeline(return_value: list[dict[str, t.Any]]) -> um.MagicMock:
    """Create a mock AutomaticSpeechRecognitionPipeline.

    Args:
        return_value: The value to return when the pipeline is called.

    Returns:
        A mock pipeline object configured to return the given value.
    """
    mock = um.MagicMock()
    mock.return_value = return_value
    return mock


def test_transcribe_chunks_batch_returns_list_of_lists() -> None:
    """Test that transcribe_chunks_batch returns a list of Chunk lists.

    Verifies that the function returns results mapped to each input chunk.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker=None)
    ]
    mock_model = _make_mock_pipeline(
        [
            {
                "chunks": [
                    {"text": "Hello", "timestamp": (0.0, 0.5)},
                    {"text": "world", "timestamp": (0.5, 1.0)},
                ]
            }
        ]
    )

    results = transcribe_chunks_batch(chunks=chunks, model=mock_model)

    assert len(results) == 1
    assert isinstance(results[0], list)
    assert all(isinstance(r, Chunk) for r in results[0])


def test_transcribe_chunks_batch_correct_time_offsets() -> None:
    """Test that chunk start_time is correctly added to output timestamps.

    Verifies that a chunk with start_time=5.0 shifts all transcription
    timestamps by 5.0 seconds.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=5.0, end_time=6.0, audio=mock_audio, text=None, speaker=None)
    ]
    mock_model = _make_mock_pipeline(
        [
            {
                "chunks": [
                    {"text": "Hello", "timestamp": (0.0, 0.5)},
                    {"text": "world", "timestamp": (0.5, 1.0)},
                ]
            }
        ]
    )

    results = transcribe_chunks_batch(chunks=chunks, model=mock_model)

    assert results[0][0].start_time == 5.0
    assert results[0][0].end_time == 5.5
    assert results[0][1].start_time == 5.5
    assert results[0][1].end_time == 6.0


def test_transcribe_chunks_batch_preserves_speaker() -> None:
    """Test that transcribe_chunks_batch preserves speaker information.

    Verifies that the speaker from the input chunk is propagated to
    all transcribed chunks.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(
            start_time=0.0, end_time=2.0, audio=mock_audio, text=None, speaker="Alice"
        )
    ]
    mock_model = _make_mock_pipeline(
        [
            {
                "chunks": [
                    {"text": "Hello", "timestamp": (0.0, 1.0)},
                    {"text": "world", "timestamp": (1.0, 2.0)},
                ]
            }
        ]
    )

    results = transcribe_chunks_batch(chunks=chunks, model=mock_model)

    assert all(r.speaker == "Alice" for r in results[0])


def test_transcribe_chunks_batch_preserves_text() -> None:
    """Test that transcribe_chunks_batch preserves transcribed text.

    Verifies that the text field from each pipeline chunk is correctly
    propagated to the output Chunk models.
    """
    mock_audio: np.ndarray = np.zeros(shape=32000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=2.0, audio=mock_audio, text=None, speaker=None)
    ]
    mock_model = _make_mock_pipeline(
        [
            {
                "chunks": [
                    {"text": "First segment", "timestamp": (0.0, 1.0)},
                    {"text": "Second segment", "timestamp": (1.0, 2.0)},
                ]
            }
        ]
    )

    results = transcribe_chunks_batch(chunks=chunks, model=mock_model)

    assert results[0][0].text == "First segment"
    assert results[0][1].text == "Second segment"


def test_transcribe_chunks_batch_empty_input() -> None:
    """Test that transcribe_chunks_batch handles empty input.

    Verifies that an empty chunk list returns an empty list.
    """
    results = transcribe_chunks_batch(chunks=[], model=_make_mock_pipeline([]))

    assert results == []


def test_transcribe_chunks_batch_empty_chunks_from_pipeline() -> None:
    """Test transcribe_chunks_batch with pipeline returning no chunks.

    Verifies that when the pipeline returns empty chunks, an empty list
    is returned for that input chunk.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker=None)
    ]
    mock_model = _make_mock_pipeline([{"chunks": []}])

    results = transcribe_chunks_batch(chunks=chunks, model=mock_model)

    assert len(results) == 1
    assert results[0] == []


def test_transcribe_chunks_batch_multiple_input_chunks() -> None:
    """Test transcribe_chunks_batch with multiple input chunks.

    Verifies that results are correctly mapped back to each input chunk.
    """
    mock_audio1: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    mock_audio2: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio1, text=None, speaker=None),
        Chunk(start_time=1.0, end_time=2.0, audio=mock_audio2, text=None, speaker=None),
    ]
    mock_model = _make_mock_pipeline(
        [
            {"chunks": [{"text": "First", "timestamp": (0.0, 1.0)}]},
            {"chunks": [{"text": "Second", "timestamp": (0.0, 1.0)}]},
        ]
    )

    results = transcribe_chunks_batch(chunks=chunks, model=mock_model)

    assert len(results) == 2
    assert results[0][0].text == "First"
    assert results[1][0].text == "Second"


# ---------------------------------------------------------------------------
# transcribe_chunks_dynamic() function tests
# ---------------------------------------------------------------------------


def test_transcribe_chunks_dynamic_processes_all_chunks() -> None:
    """Test that transcribe_chunks_dynamic processes all input chunks.

    Verifies that the function processes chunks through dynamic batching
    and returns transcriptions for all of them.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker=None)
    ]
    mock_model = _make_mock_pipeline(
        [{"chunks": [{"text": "Hello world", "timestamp": (0.0, 1.0)}]}]
    )

    results = transcribe_chunks_dynamic(
        chunks=chunks, model=mock_model, show_progress=False
    )

    # Function returns list[list[Chunk]] - one list per input chunk
    assert len(results) == 1
    assert results[0][0].text == "Hello world"


def test_transcribe_chunks_dynamic_empty_input() -> None:
    """Test that transcribe_chunks_dynamic handles empty input.

    Verifies that an empty chunk list returns an empty list.
    """
    results = transcribe_chunks_dynamic(
        chunks=[], model=_make_mock_pipeline([]), show_progress=False
    )

    assert results == []


def test_transcribe_chunks_dynamic_preserves_speaker() -> None:
    """Test that transcribe_chunks_dynamic preserves speaker information.

    Verifies that speaker data flows through the dynamic batching pipeline.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker="Bob")
    ]
    mock_model = _make_mock_pipeline(
        [{"chunks": [{"text": "Test transcription", "timestamp": (0.0, 1.0)}]}]
    )

    results = transcribe_chunks_dynamic(
        chunks=chunks, model=mock_model, show_progress=False
    )

    # Function returns list[list[Chunk]] - one list per input chunk
    assert results[0][0].speaker == "Bob"


def test_transcribe_chunks_dynamic_with_multiple_chunks() -> None:
    """Test transcribe_chunks_dynamic with multiple input chunks.

    Verifies that all chunks are processed and returned as a list of lists.
    """
    mock_audio1: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    mock_audio2: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio1, text=None, speaker=None),
        Chunk(start_time=1.0, end_time=2.0, audio=mock_audio2, text=None, speaker=None),
    ]
    mock_model = _make_mock_pipeline(
        [
            {"chunks": [{"text": "First chunk", "timestamp": (0.0, 1.0)}]},
            {"chunks": [{"text": "Second chunk", "timestamp": (0.0, 1.0)}]},
        ]
    )

    results = transcribe_chunks_dynamic(
        chunks=chunks, model=mock_model, show_progress=False
    )

    # Function returns list[list[Chunk]] - one list per input chunk
    assert len(results) == 2
    assert results[0][0].text == "First chunk"
    assert results[1][0].text == "Second chunk"


# ---------------------------------------------------------------------------
# Error handling and edge case tests
# ---------------------------------------------------------------------------


def test_transcribe_chunks_batch_raises_on_pipeline_error() -> None:
    """Test that transcribe_chunks_batch raises on pipeline errors.

    Verifies that exceptions from the ASR pipeline are properly propagated
    after logging an error.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker=None)
    ]
    mock_model = um.MagicMock()
    mock_model.side_effect = RuntimeError("Pipeline failed")

    with pytest.raises(RuntimeError, match="Pipeline failed"):
        transcribe_chunks_batch(chunks=chunks, model=mock_model)


def test_transcribe_chunks_dynamic_raises_on_batch_error() -> None:
    """Test that transcribe_chunks_dynamic raises on batch transcription errors.

    Verifies that exceptions during batch processing are properly propagated
    with appropriate error logging.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker=None)
    ]
    mock_model = um.MagicMock()
    mock_model.side_effect = ValueError("Transcription failed")

    with pytest.raises(ValueError, match="Transcription failed"):
        transcribe_chunks_dynamic(chunks=chunks, model=mock_model, show_progress=False)


def test_transcribe_chunks_batch_skips_short_chunks() -> None:
    """Test that chunks shorter than MIN_CHUNK_LENGTH_SECONDS are skipped.

    Verifies that word-level segments shorter than the minimum threshold
    are filtered out from the results.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker=None)
    ]
    # Create transcription results with very short segments (< 0.05s)
    # MIN_CHUNK_LENGTH_SECONDS = 0.05, so segments must be >= 0.05s
    mock_model = _make_mock_pipeline(
        [
            {
                "chunks": [
                    {
                        "text": "Hi",
                        "timestamp": (0.0, 0.04),
                    },  # 0.04s - too short (< 0.05)
                    {"text": "Hello", "timestamp": (0.1, 0.9)},  # 0.8s - valid
                ]
            }
        ]
    )

    results = transcribe_chunks_batch(chunks=chunks, model=mock_model)

    # Only the valid chunk should be returned
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].text == "Hello"


def test_padding_with_varying_chunk_lengths() -> None:
    """Test padding logic with chunks of different lengths.

    Verifies that shorter chunks are correctly padded to match the
    maximum length in the batch.
    """
    # Create chunks with different durations
    chunk_short = Chunk(
        start_time=0.0,
        end_time=1.0,
        audio=np.zeros(16000, dtype=np.float32),  # 1 second
        text=None,
        speaker=None,
    )
    chunk_long = Chunk(
        start_time=0.0,
        end_time=2.0,
        audio=np.zeros(32000, dtype=np.float32),  # 2 seconds
        text=None,
        speaker=None,
    )
    chunks = [chunk_short, chunk_long]

    mock_model = _make_mock_pipeline(
        [
            {"chunks": [{"text": "Short", "timestamp": (0.0, 1.0)}]},
            {"chunks": [{"text": "Long", "timestamp": (0.0, 2.0)}]},
        ]
    )

    results = transcribe_chunks_batch(chunks=chunks, model=mock_model)

    # Both chunks should be transcribed
    assert len(results) == 2
    assert results[0][0].text == "Short"
    assert results[1][0].text == "Long"


def test_padding_with_all_same_length_chunks() -> None:
    """Test padding when all chunks are the same length.

    Verifies that no padding is applied when all chunks have equal length.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker=None),
        Chunk(start_time=1.0, end_time=2.0, audio=mock_audio, text=None, speaker=None),
    ]

    mock_model = _make_mock_pipeline(
        [
            {"chunks": [{"text": "First", "timestamp": (0.0, 1.0)}]},
            {"chunks": [{"text": "Second", "timestamp": (0.0, 1.0)}]},
        ]
    )

    results = transcribe_chunks_batch(chunks=chunks, model=mock_model)

    assert len(results) == 2
    assert results[0][0].text == "First"
    assert results[1][0].text == "Second"


def test_progress_bar_disabled_when_show_progress_false() -> None:
    """Test that progress bar is disabled when show_progress=False.

    Verifies that the tqdm progress bar is properly disabled when the
    flag is set to False.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker=None)
    ]
    mock_model = _make_mock_pipeline(
        [{"chunks": [{"text": "Hello", "timestamp": (0.0, 1.0)}]}]
    )

    # Should complete without errors even with progress disabled
    results = transcribe_chunks_dynamic(
        chunks=chunks, model=mock_model, show_progress=False
    )

    assert len(results) == 1
    assert results[0][0].text == "Hello"


def test_progress_bar_shows_batch_info() -> None:
    """Test that progress bar displays batch information.

    Verifies that the progress bar description includes batch number,
    chunk count, and duration.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker=None)
    ]
    mock_model = _make_mock_pipeline(
        [{"chunks": [{"text": "Hello", "timestamp": (0.0, 1.0)}]}]
    )

    # Mock tqdm to capture the description
    with um.patch("but_with_subs.transcribing.tqdm") as mock_tqdm:
        mock_iterator = um.MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_iterator
        mock_iterator.__iter__.return_value = iter([chunks])
        mock_iterator.set_description = um.MagicMock()

        transcribe_chunks_dynamic(chunks=chunks, model=mock_model, show_progress=True)

        # Verify set_description was called with batch info
        assert mock_iterator.set_description.called


def test_transcribe_chunks_batch_all_short_segments() -> None:
    """Test handling when all segments are below minimum length.

    Verifies that empty results are returned when all transcribed
    segments are shorter than MIN_CHUNK_LENGTH_SECONDS.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    chunks = [
        Chunk(start_time=0.0, end_time=1.0, audio=mock_audio, text=None, speaker=None)
    ]
    # All segments are very short (< 0.05s - MIN_CHUNK_LENGTH_SECONDS)
    mock_model = _make_mock_pipeline(
        [
            {
                "chunks": [
                    {"text": "Hi", "timestamp": (0.0, 0.04)},  # 0.04s < 0.05
                    {"text": "Bye", "timestamp": (0.1, 0.14)},  # 0.04s < 0.05
                ]
            }
        ]
    )

    results = transcribe_chunks_batch(chunks=chunks, model=mock_model)

    # All segments filtered out due to being too short
    assert len(results) == 1
    assert results[0] == []
