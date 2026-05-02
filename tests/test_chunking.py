"""Tests for the chunking module.

This module contains tests for the Chunk model.
"""

import contextlib as ct
import pathlib as pt
import tempfile as tf
import typing as t
import unittest.mock as um

import numpy as np
import scipy.io.wavfile as wavio

from but_with_subs.audio_chunking import (
    _load_audio,
    _split_audio_into_chunks,
    chunk_audio,
)
from but_with_subs.data_models import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@ct.contextmanager
def _temp_wav_file() -> t.Generator[pt.Path, None, None]:
    """Context manager that yields a temporary WAV file path.

    Creates a temporary file and yields its path. The file is
    automatically cleaned up when the context exits.

    Yields:
        A pathlib.Path to the temporary WAV file.
    """
    with tf.NamedTemporaryFile(suffix=".wav", delete=True, dir=".") as tmp:
        yield pt.Path(tmp.name)


# ---------------------------------------------------------------------------
# Chunk model tests
# ---------------------------------------------------------------------------


def test_chunk_model_creation_with_all_fields() -> None:
    """Test constructing a Chunk model with all fields populated."""
    audio_data: np.ndarray = np.zeros(shape=16000, dtype=np.float64)

    chunk: Chunk = Chunk(start_time=0.0, end_time=1.0, audio=audio_data)

    assert chunk.start_time == 0.0
    assert chunk.end_time == 1.0
    np.testing.assert_array_equal(chunk.audio, audio_data)


def test_chunk_model_with_nonzero_start() -> None:
    """Test constructing a Chunk model with a non-zero start time."""
    audio_data: np.ndarray = np.zeros(shape=8000, dtype=np.float64)

    chunk = Chunk(start_time=1.5, end_time=3.0, audio=audio_data)

    assert chunk.start_time == 1.5
    assert chunk.end_time == 3.0
    assert chunk.audio.shape == (8000,)


def test_chunk_model_duration_matches_audio_length() -> None:
    """Test that chunk duration matches the audio data length."""
    num_samples: int = 32000
    audio_data: np.ndarray = np.zeros(shape=num_samples, dtype=np.float64)

    chunk = Chunk(start_time=0.0, end_time=2.0, audio=audio_data)

    assert chunk.end_time - chunk.start_time == 2.0
    assert chunk.audio.size == num_samples


# ---------------------------------------------------------------------------
# _load_audio tests
# ---------------------------------------------------------------------------


def test_load_audio_returns_correct_sample_rate() -> None:
    """Test _load_audio returns the correct sample rate from a WAV file."""
    with _temp_wav_file() as temp_path:
        wavio.write(
            filename=temp_path, rate=22050, data=np.zeros(shape=1000, dtype=np.int16)
        )
        sr, _ = _load_audio(path=temp_path)

    assert sr == 22050


def test_load_audio_returns_mono_from_stereo() -> None:
    """Test _load_audio converts stereo audio to mono."""
    with _temp_wav_file() as temp_path:
        stereo_data = np.stack(
            [
                np.sin(np.linspace(0, 100 * np.pi, 1000)),
                np.sin(np.linspace(0, 200 * np.pi, 1000)),
            ],
            axis=1,
        ).astype(np.int16)
        wavio.write(filename=temp_path, rate=16000, data=stereo_data)
        sr, audio = _load_audio(path=temp_path)

    assert sr == 16000
    assert audio.ndim == 1


def test_load_audio_raises_on_empty_file() -> None:
    """Test _load_audio raises ValueError for empty audio files."""
    with _temp_wav_file() as temp_path:
        wavio.write(
            filename=temp_path, rate=16000, data=np.zeros(shape=0, dtype=np.int16)
        )

        try:
            _load_audio(path=temp_path)
        except ValueError as e:
            assert "no data" in str(e).lower()
        else:
            assert False, "Expected ValueError to be raised"


# ---------------------------------------------------------------------------
# _split_audio_into_chunks tests
# ---------------------------------------------------------------------------


def test_split_audio_into_chunks_with_speech_returns_chunks() -> None:
    """Test _split_audio_into_chunks returns chunks for speech audio.

    Mocks get_speech_timestamps to return speech segments and verifies
    that chunk segments are correctly extracted from the audio array.
    """
    mock_audio = np.zeros(shape=32000, dtype=np.float64)

    mock_timestamps = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]

    with (
        um.patch(
            "but_with_subs.chunking.get_speech_timestamps", return_value=mock_timestamps
        ),
        um.patch("but_with_subs.chunking.load_silero_vad", return_value=None),
    ):
        chunks = _split_audio_into_chunks(audio=mock_audio)

    assert len(chunks) == 2
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 1.0
    assert chunks[0].audio.shape == (16000,)
    assert chunks[1].start_time == 2.0
    assert chunks[1].end_time == 3.0


def test_split_audio_into_chunks_with_no_speech_returns_empty() -> None:
    """Test _split_audio_into_chunks returns empty list for silent audio."""
    mock_audio = np.zeros(shape=16000, dtype=np.float64)

    with (
        um.patch("but_with_subs.chunking.get_speech_timestamps", return_value=[]),
        um.patch("but_with_subs.chunking.load_silero_vad", return_value=None),
    ):
        chunks = _split_audio_into_chunks(audio=mock_audio)

    assert chunks == []


# ---------------------------------------------------------------------------
# chunk_audio integration tests
# ---------------------------------------------------------------------------


def test_chunk_audio_returns_chunks() -> None:
    """Test chunk_audio returns Chunk models when speech timestamps exist."""
    mock_sr = 16000
    mock_audio = np.zeros(shape=16000, dtype=np.float64)

    mock_timestamps = [{"start": 0.0, "end": 1.0}]

    with (
        um.patch(
            "but_with_subs.chunking._load_audio", return_value=(mock_sr, mock_audio)
        ),
        um.patch(
            "but_with_subs.chunking._resample_to_16k_mono", return_value=mock_audio
        ),
        um.patch(
            "but_with_subs.chunking.get_speech_timestamps", return_value=mock_timestamps
        ),
        um.patch("but_with_subs.chunking.load_silero_vad", return_value=None),
    ):
        chunks = list(chunk_audio(audio_path=pt.Path("/fake/audio.wav")))

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
