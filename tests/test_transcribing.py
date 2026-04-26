"""Tests for the transcribing module.

This module contains comprehensive tests for the Transcription model and the
transcribe function, including mocking the AutomaticSpeechRecognitionPipeline
to verify correct behavior under various conditions.
"""

import typing as t
import unittest.mock as um

import numpy as np

from but_with_subs.transcribing import Transcription, transcribe

# ---------------------------------------------------------------------------
# Transcription model tests
# ---------------------------------------------------------------------------


def test_transcription_model_creation_with_all_fields() -> None:
    """Test constructing a Transcription model with all fields populated.

    Verifies that the Transcription model can be instantiated with
    start_time, end_time, and text, and that the values are stored
    correctly.
    """
    transcription: Transcription = Transcription(
        start_time=0.0, end_time=1.5, text="Hello world"
    )

    assert transcription.start_time == 0.0
    assert transcription.end_time == 1.5
    assert transcription.text == "Hello world"


def test_transcription_model_with_nonzero_start() -> None:
    """Test constructing a Transcription model with a non-zero start time.

    Verifies that the Transcription model correctly stores a start_time
    greater than zero, representing a segment that begins after the audio
    start.
    """
    transcription = Transcription(
        start_time=2.5, end_time=5.0, text="This is a later segment"
    )

    assert transcription.start_time == 2.5
    assert transcription.end_time == 5.0
    assert transcription.text == "This is a later segment"


def test_transcription_duration_matches() -> None:
    """Test that duration equals end_time minus start_time.

    Verifies that for a Transcription with start_time=0.0 and
    end_time=3.0, the duration (end_time - start_time) equals 3.0.
    """
    transcription = Transcription(
        start_time=0.0, end_time=3.0, text="Three second segment"
    )

    assert transcription.end_time - transcription.start_time == 3.0


def test_transcription_duration_with_offset() -> None:
    """Test duration is correct when start_time is non-zero.

    Verifies that the duration calculation works correctly when
    start_time is greater than zero.
    """
    transcription = Transcription(
        start_time=1.5, end_time=4.5, text="Three second segment with offset"
    )

    assert transcription.end_time - transcription.start_time == 3.0


# ---------------------------------------------------------------------------
# transcribe() function tests
# ---------------------------------------------------------------------------


def _make_mock_pipeline(return_value: dict[str, t.Any]) -> um.MagicMock:
    """Create a mock AutomaticSpeechRecognitionPipeline.

    Args:
        return_value: The value to return when the pipeline is called.

    Returns:
        A mock pipeline object configured to return the given value.
    """
    mock = um.MagicMock()
    mock.return_value = return_value
    return mock


def test_transcribe_yields_transcription_models() -> None:
    """Test that transcribe returns Transcription models.

    Mocks the pipeline to return known transcription chunks and
    verifies that the returned list contains Transcription instances.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    mock_model = _make_mock_pipeline(
        return_value={
            "chunks": [
                {"text": "Hello", "timestamp": (0.0, 0.5)},
                {"text": "world", "timestamp": (0.5, 1.0)},
            ]
        }
    )

    results = transcribe(audio_data=mock_audio, model=mock_model, chunk_offset=0.0)

    assert len(results) == 2
    assert all(isinstance(r, Transcription) for r in results)


def test_transcribe_correct_time_offsets() -> None:
    """Test that chunk_offset is correctly added to start/end times.

    Mocks the pipeline to return chunks with timestamps starting at 0.0
    and verifies that a chunk_offset of 5.0 shifts all times by 5.0.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    mock_model = _make_mock_pipeline(
        return_value={
            "chunks": [
                {"text": "Hello", "timestamp": (0.0, 0.5)},
                {"text": "world", "timestamp": (0.5, 1.0)},
            ]
        }
    )

    results = transcribe(audio_data=mock_audio, model=mock_model, chunk_offset=5.0)

    assert results[0].start_time == 5.0
    assert results[0].end_time == 5.5
    assert results[1].start_time == 5.5
    assert results[1].end_time == 6.0


def test_transcribe_preserves_text() -> None:
    """Test that transcribe preserves the transcribed text from the pipeline.

    Verifies that the text field from each pipeline chunk is correctly
    propagated to the Transcription model.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    mock_model = _make_mock_pipeline(
        return_value={
            "chunks": [
                {"text": "First segment", "timestamp": (0.0, 1.0)},
                {"text": "Second segment", "timestamp": (1.0, 2.0)},
            ]
        }
    )

    results = transcribe(audio_data=mock_audio, model=mock_model, chunk_offset=0.0)

    assert results[0].text == "First segment"
    assert results[1].text == "Second segment"


def test_transcribe_empty_audio_empty_result() -> None:
    """Test that transcribe returns an empty list for empty pipeline output.

    Mocks the pipeline to return a result with no chunks and verifies
    that an empty list is returned.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    mock_model = _make_mock_pipeline(return_value={"chunks": []})

    results = transcribe(audio_data=mock_audio, model=mock_model, chunk_offset=0.0)

    assert results == []


def test_transcribe_multiple_chunks_from_pipeline() -> None:
    """Test transcribe handles multiple chunks from the pipeline.

    Mocks the pipeline to return multiple chunks and verifies that
    each chunk is converted to a Transcription model.
    """
    mock_audio: np.ndarray = np.zeros(shape=32000, dtype=np.float32)
    mock_model = _make_mock_pipeline(
        return_value={
            "chunks": [
                {"text": "A", "timestamp": (0.0, 0.2)},
                {"text": "B", "timestamp": (0.2, 0.4)},
                {"text": "C", "timestamp": (0.4, 0.6)},
            ]
        }
    )

    results = transcribe(audio_data=mock_audio, model=mock_model, chunk_offset=0.0)

    assert len(results) == 3
    assert results[0].text == "A"
    assert results[1].text == "B"
    assert results[2].text == "C"


def test_transcribe_duration_matches_timestamps() -> None:
    """Test that Transcription duration matches the pipeline timestamps.

    Verifies that the duration (end_time - start_time) of each returned
    Transcription matches the difference between the timestamp pair
    from the pipeline.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    mock_model = _make_mock_pipeline(
        return_value={"chunks": [{"text": "Segment", "timestamp": (1.0, 3.5)}]}
    )

    results = transcribe(audio_data=mock_audio, model=mock_model, chunk_offset=0.0)

    assert len(results) == 1
    assert results[0].end_time - results[0].start_time == 2.5


def test_transcribe_with_nonzero_offset_preserves_duration() -> None:
    """Test that chunk_offset does not affect segment duration.

    Verifies that adding a chunk_offset shifts both start and end times
    equally, preserving the duration of each segment.
    """
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float32)
    mock_model = _make_mock_pipeline(
        return_value={"chunks": [{"text": "Segment", "timestamp": (0.5, 2.0)}]}
    )

    results = transcribe(audio_data=mock_audio, model=mock_model, chunk_offset=10.0)

    assert len(results) == 1
    assert results[0].start_time == 10.5
    assert results[0].end_time == 12.0
    assert results[0].end_time - results[0].start_time == 1.5
