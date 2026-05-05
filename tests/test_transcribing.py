"""Tests for the transcribing module.

This module contains tests for the transcribe_audio function,
including mocking the AutomaticSpeechRecognitionPipeline to verify
correct behavior under various conditions.
"""

import unittest.mock as um

import numpy as np
import pytest

from but_with_subs.constants import MIN_CHUNK_LENGTH_SECONDS
from but_with_subs.data_models import Chunk
from but_with_subs.transcribing import transcribe_audio

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_audio() -> np.ndarray:
    """Create a 2-second audio array at 16 kHz.

    Returns:
        A NumPy array of float32 samples.
    """
    duration = 2.0
    sample_rate = 16_000
    n_samples = int(duration * sample_rate)
    return np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)).astype(
        np.float32
    )


@pytest.fixture
def mock_model_with_chunks() -> um.MagicMock:
    """Create a mock ASR pipeline returning word-level chunks.

    Returns:
        A MagicMock configured as a mock pipeline.
    """
    mock = um.MagicMock()
    mock.return_value = {
        "chunks": [
            {"text": "Hello", "timestamp": (0.0, 0.5)},
            {"text": "world", "timestamp": (0.5, 1.0)},
            {"text": "this", "timestamp": (1.0, 1.5)},
            {"text": "is", "timestamp": (1.5, 2.0)},
        ]
    }
    return mock


# ---------------------------------------------------------------------------
# transcribe_audio() function tests
# ---------------------------------------------------------------------------


def test_transcribe_audio_returns_list_of_chunks(mock_audio: np.ndarray) -> None:
    """Test that transcribe_audio returns a list of Chunk objects."""
    mock_model = um.MagicMock()
    mock_model.return_value = {
        "chunks": [
            {"text": "Hello", "timestamp": (0.0, 0.5)},
            {"text": "world", "timestamp": (0.5, 1.0)},
        ]
    }

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    assert isinstance(results, list)
    assert all(isinstance(r, Chunk) for r in results)


def test_transcribe_audio_correct_time_offsets(mock_audio: np.ndarray) -> None:
    """Test that timestamps are correctly extracted from pipeline output."""
    mock_model = um.MagicMock()
    mock_model.return_value = {
        "chunks": [
            {"text": "Hello", "timestamp": (0.0, 0.5)},
            {"text": "world", "timestamp": (0.5, 1.0)},
        ]
    }

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    assert results[0].start_time == 0.0
    assert results[0].end_time == 0.5
    assert results[1].start_time == 0.5
    assert results[1].end_time == 1.0


def test_transcribe_audio_preserves_text(mock_audio: np.ndarray) -> None:
    """Test that transcribed text is correctly propagated."""
    mock_model = um.MagicMock()
    mock_model.return_value = {
        "chunks": [
            {"text": "First segment", "timestamp": (0.0, 1.0)},
            {"text": "Second segment", "timestamp": (1.0, 2.0)},
        ]
    }

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    assert results[0].text == "First segment"
    assert results[1].text == "Second segment"


def test_transcribe_audio_sets_speaker_to_none(mock_audio: np.ndarray) -> None:
    """Test that speaker is set to None since no diarization is done."""
    mock_model = um.MagicMock()
    mock_model.return_value = {"chunks": [{"text": "Hello", "timestamp": (0.0, 0.5)}]}

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    assert results[0].speaker is None


def test_transcribe_audio_empty_input(mock_audio: np.ndarray) -> None:
    """Test that empty pipeline output returns an empty list."""
    mock_model = um.MagicMock()
    mock_model.return_value = {"chunks": []}

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    assert results == []


def test_transcribe_audio_skips_short_segments(mock_audio: np.ndarray) -> None:
    """Test that segments shorter than MIN_CHUNK_LENGTH_SECONDS are skipped."""
    mock_model = um.MagicMock()
    mock_model.return_value = {
        "chunks": [
            {"text": "Hi", "timestamp": (0.0, 0.04)},  # 0.04s < 0.05
            {"text": "Hello", "timestamp": (0.1, 0.9)},  # 0.8s - valid
        ]
    }

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    assert len(results) == 1
    assert results[0].text == "Hello"


def test_transcribe_audio_custom_min_chunk_length(mock_audio: np.ndarray) -> None:
    """Test that a custom min_chunk_length threshold works."""
    mock_model = um.MagicMock()
    mock_model.return_value = {
        "chunks": [
            {"text": "Short", "timestamp": (0.0, 0.3)},  # 0.3s
            {"text": "Long", "timestamp": (0.3, 1.5)},  # 1.2s
        ]
    }

    # With threshold 1.0, only the 1.2s segment should remain
    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(
            audio=mock_audio, model=mock_model, min_chunk_length=1.0, show_progress=False
        )

    assert len(results) == 1
    assert results[0].text == "Long"


def test_transcribe_audio_all_short_segments_filtered(mock_audio: np.ndarray) -> None:
    """Test that all-short-segments results in empty list."""
    mock_model = um.MagicMock()
    mock_model.return_value = {
        "chunks": [
            {"text": "Hi", "timestamp": (0.0, 0.04)},
            {"text": "Bye", "timestamp": (0.1, 0.14)},
        ]
    }

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    assert results == []


def test_transcribe_audio_audio_slice_correct(mock_audio: np.ndarray) -> None:
    """Test that audio slices match the segment timestamps."""
    sample_rate = 16_000
    mock_model = um.MagicMock()
    mock_model.return_value = {"chunks": [{"text": "Hello", "timestamp": (0.5, 1.0)}]}

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    expected_start = int(0.5 * sample_rate)
    expected_end = int(1.0 * sample_rate)
    expected_audio = mock_audio[expected_start:expected_end]

    np.testing.assert_array_equal(results[0].audio, expected_audio)


def test_transcribe_audio_audio_clipped_to_audio_length(mock_audio: np.ndarray) -> None:
    """Test that audio slicing respects the actual audio length."""
    sample_rate = 16_000
    # Create a 1-second audio array
    short_audio = mock_audio[: int(sample_rate * 1.0)]

    mock_model = um.MagicMock()
    # Pipeline reports a segment past the end of the audio
    mock_model.return_value = {"chunks": [{"text": "Hello", "timestamp": (0.5, 2.0)}]}

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 1.0, short_audio)]):
        results = transcribe_audio(audio=short_audio, model=mock_model, show_progress=False)

    # Audio should be clipped to the actual array length
    expected_audio = short_audio[int(0.5 * sample_rate) :]
    np.testing.assert_array_equal(results[0].audio, expected_audio)


def test_progress_bar_disabled_when_show_progress_false(mock_audio: np.ndarray) -> None:
    """Test that progress bar is disabled when show_progress=False."""
    mock_model = um.MagicMock()
    mock_model.return_value = {"chunks": [{"text": "Hello", "timestamp": (0.0, 1.0)}]}

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        # Should complete without errors
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    assert len(results) == 1
    assert results[0].text == "Hello"


# ---------------------------------------------------------------------------
# Error handling and edge cases
# ---------------------------------------------------------------------------


def test_transcribe_audio_handles_pipeline_error(mock_audio: np.ndarray) -> None:
    """Test that pipeline errors are logged and return empty results."""
    mock_model = um.MagicMock()
    mock_model.side_effect = RuntimeError("Pipeline failed")

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    # Errors are caught per-segment and logged, returning empty results
    assert results == []


def test_transcribe_audio_all_segments_same_duration(mock_audio: np.ndarray) -> None:
    """Test with uniformly sized segments."""
    mock_model = um.MagicMock()
    mock_model.return_value = {
        "chunks": [
            {"text": "A", "timestamp": (0.0, 0.5)},
            {"text": "B", "timestamp": (0.5, 1.0)},
            {"text": "C", "timestamp": (1.0, 1.5)},
            {"text": "D", "timestamp": (1.5, 2.0)},
        ]
    }

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    assert len(results) == 4
    texts = [r.text for r in results]
    assert texts == ["A", "B", "C", "D"]


def test_transcribe_audio_segment_at_exact_min_threshold(
    mock_audio: np.ndarray,
) -> None:
    """Test that a segment exactly at MIN_CHUNK_LENGTH_SECONDS is included."""
    mock_model = um.MagicMock()
    mock_model.return_value = {
        "chunks": [
            {"text": "Exact", "timestamp": (0.0, MIN_CHUNK_LENGTH_SECONDS)},
            {"text": "TooShort", "timestamp": (0.1, 0.14)},
        ]
    }

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    # Exact threshold should be included
    assert len(results) == 1
    assert results[0].text == "Exact"


def test_transcribe_audio_chunk_has_all_required_fields(mock_audio: np.ndarray) -> None:
    """Test that returned Chunks have all required fields set."""
    mock_model = um.MagicMock()
    mock_model.return_value = {"chunks": [{"text": "Hello", "timestamp": (0.0, 1.0)}]}

    with um.patch("but_with_subs.transcribing.vad_segment_audio", return_value=[(0.0, 2.0, mock_audio)]):
        results = transcribe_audio(audio=mock_audio, model=mock_model, show_progress=False)

    chunk = results[0]
    assert chunk.start_time is not None
    assert chunk.end_time is not None
    assert chunk.audio is not None
    assert chunk.text == "Hello"
    assert chunk.speaker is None
