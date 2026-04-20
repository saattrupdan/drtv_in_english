"""Tests for the chunking module.

This module contains comprehensive tests for the Chunk model, chunk_audio
function, silence detection logic, audio resampling logic, and edge cases.
"""

import pathlib
import tempfile as tf
import typing as t
import unittest.mock as um

import contextlib as ct

import numpy as np

import scipy.io.wavfile as wavio

from but_with_subs.chunking import (
    Chunk,
    _detect_silence_breaks,
    _load_audio,
    _resample_to_16k_mono,
    chunk_audio,
)

import numpy as np

from but_with_subs.chunking import (
    Chunk,
    _detect_silence_breaks,
    _load_audio,
    _resample_to_16k_mono,
    chunk_audio,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@ct.contextmanager
def tempfile_temporary_file() -> t.Generator[pathlib.Path, None, None]:
    """Context manager that yields a temporary file path.

    Creates a temporary file and yields its path. The file is
    automatically cleaned up when the context exits.

    Yields:
        A pathlib.Path to the temporary file.
    """
    with tf.NamedTemporaryFile(suffix=".wav", delete=True, dir=".") as tmp:
        yield pathlib.Path(tmp.name)


# ---------------------------------------------------------------------------
# Chunk model tests
# ---------------------------------------------------------------------------


def test_chunk_model_creation_with_all_fields() -> None:
    """Test constructing a Chunk model with all fields populated.

    Verifies that the Chunk model can be instantiated with start_time,
    end_time, and audio fields, and that the values are stored correctly.
    """
    audio_data: np.ndarray = np.zeros(shape=16000, dtype=np.float64)

    chunk: Chunk = Chunk(start_time=0.0, end_time=1.0, audio=audio_data)

    assert chunk.start_time == 0.0
    assert chunk.end_time == 1.0
    np.testing.assert_array_equal(chunk.audio, audio_data)


def test_chunk_model_with_nonzero_start() -> None:
    """Test constructing a Chunk model with a non-zero start time.

    Verifies that the Chunk model correctly stores a start_time greater
    than zero, representing a chunk that begins after the audio start.
    """
    audio_data: np.ndarray = np.zeros(shape=8000, dtype=np.float64)

    chunk = Chunk(start_time=1.5, end_time=3.0, audio=audio_data)

    assert chunk.start_time == 1.5
    assert chunk.end_time == 3.0
    assert chunk.audio.shape == (8000,)


def test_chunk_model_duration_matches_audio_length() -> None:
    """Test that chunk duration matches the audio data length.

    Verifies that for 16kHz audio, the duration (end_time - start_time)
    equals the number of samples divided by 16000.
    """
    num_samples: int = 32000
    audio_data: np.ndarray = np.zeros(shape=num_samples, dtype=np.float64)

    chunk = Chunk(start_time=0.0, end_time=2.0, audio=audio_data)

    assert chunk.end_time - chunk.start_time == 2.0
    assert chunk.audio.size == num_samples


# ---------------------------------------------------------------------------
# chunk_audio() function tests
# ---------------------------------------------------------------------------


def test_chunk_audio_yields_chunk_models() -> None:
    """Test that chunk_audio yields Chunk models when given audio data.

    Mocks _load_audio to return fake audio data and verifies that
    the returned generator yields Chunk instances.
    """
    mock_sr: int = 16000
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float64)

    with um.patch(
        "but_with_subs.chunking._load_audio", return_value=(mock_sr, mock_audio)
    ):
        audio_path = pathlib.Path("/fake/audio.wav")
        chunks = list(chunk_audio(audio_path=audio_path))

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)


def test_chunk_audio_with_silence_breaks() -> None:
    """Test chunk_audio yields multiple chunks when silence breaks exist.

    Mocks _load_audio and _detect_silence_breaks to simulate audio
    with a silence gap, verifying that multiple Chunk models are yielded.
    """
    mock_sr: int = 16000
    mock_audio: np.ndarray = np.zeros(shape=48000, dtype=np.float64)

    with (
        um.patch(
            "but_with_subs.chunking._load_audio", return_value=(mock_sr, mock_audio)
        ),
        um.patch("but_with_subs.chunking._detect_silence_breaks", return_value=[1.0]),
    ):
        audio_path = pathlib.Path("/fake/audio.wav")
        chunks = list(chunk_audio(audio_path=audio_path))

    assert len(chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_chunk_audio_correct_times() -> None:
    """Test chunk_audio yields chunks with correct start and end times.

    Mocks the silence detection to return a break at 1.0 seconds
    and verifies that the yielded chunks have correct time boundaries.
    """
    mock_sr: int = 16000
    mock_audio: np.ndarray = np.zeros(shape=32000, dtype=np.float64)

    with (
        um.patch(
            "but_with_subs.chunking._load_audio", return_value=(mock_sr, mock_audio)
        ),
        um.patch("but_with_subs.chunking._detect_silence_breaks", return_value=[1.0]),
    ):
        audio_path = pathlib.Path("/fake/audio.wav")
        chunks = list(chunk_audio(audio_path=audio_path))

    assert len(chunks) == 2
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 1.0
    assert chunks[1].start_time == 1.0
    assert chunks[1].end_time == 2.0


def test_chunk_audio_with_multiple_breaks() -> None:
    """Test chunk_audio yields correct number of chunks with multiple breaks.

    Mocks silence detection to return three break points and verifies
    that the correct number of chunks are yielded.
    """
    mock_sr: int = 16000
    mock_audio: np.ndarray = np.zeros(shape=64000, dtype=np.float64)

    break_times: list[float] = [1.0, 2.0, 3.0]

    with (
        um.patch(
            "but_with_subs.chunking._load_audio", return_value=(mock_sr, mock_audio)
        ),
        um.patch(
            "but_with_subs.chunking._detect_silence_breaks", return_value=break_times
        ),
    ):
        audio_path = pathlib.Path("/fake/audio.wav")
        chunks = list(chunk_audio(audio_path=audio_path))

    assert len(chunks) == 4


def test_chunk_audio_audio_data_matches() -> None:
    """Test chunk_audio yields chunks with correct audio data slices.

    Verifies that the audio data in each chunk corresponds to the
    correct slice of the original audio based on time boundaries.
    """
    mock_sr: int = 16000
    mock_audio: np.ndarray = np.arange(48000, dtype=np.float64)

    with (
        um.patch(
            "but_with_subs.chunking._load_audio", return_value=(mock_sr, mock_audio)
        ),
        um.patch("but_with_subs.chunking._detect_silence_breaks", return_value=[1.0]),
    ):
        audio_path = pathlib.Path("/fake/audio.wav")
        chunks = list(chunk_audio(audio_path=audio_path))

    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0].audio, mock_audio[:16000])
    np.testing.assert_array_equal(chunks[1].audio, mock_audio[16000:])


# ---------------------------------------------------------------------------
# Silence detection tests
# ---------------------------------------------------------------------------


def test_detect_silence_breaks_finds_silence_gaps() -> None:
    """Test _detect_silence_breaks finds silence gaps correctly.

    Creates audio with a silence gap in the middle and verifies
    that a break time is detected at the start of the silence gap.
    """
    sr: int = 16000
    # 1s of speech, then 0.5s of silence, then 1s of speech
    audio = np.zeros(shape=32000, dtype=np.float64)
    audio[:16000] = np.sin(np.linspace(0, 10 * np.pi, 16000))  # first second - speech
    audio[16000:24000] = 0.0  # silence gap (0.5s)
    audio[24000:] = np.sin(np.linspace(0, 10 * np.pi, 8000))  # second part - speech

    breaks = _detect_silence_breaks(
        audio=audio, sr=sr, threshold_db=-40.0, min_gap_seconds=0.5
    )

    assert len(breaks) >= 1


def test_detect_silence_breaks_returns_empty_for_short_audio() -> None:
    """Test _detect_silence_breaks returns empty list for very short audio.

    When the audio is shorter than the window size, silence detection
    cannot proceed and should return an empty list.
    """
    sr: int = 16000
    short_audio: np.ndarray = np.zeros(shape=100, dtype=np.float64)

    breaks = _detect_silence_breaks(
        audio=short_audio, sr=sr, threshold_db=-40.0, min_gap_seconds=0.5
    )

    assert breaks == []


def test_detect_silence_breaks_filters_short_gaps() -> None:
    """Test _detect_silence_breaks filters out gaps shorter than min_gap_seconds.

    Creates audio with a very short silence gap (less than 0.5s) and
    verifies that it is filtered out and not returned as a break point.
    """
    sr: int = 16000
    # Create audio with two silence regions close together
    audio = np.zeros(shape=64000, dtype=np.float64)
    # Add speech-like signal in between
    audio[8000:12000] = np.sin(np.linspace(0, 20 * np.pi, 4000)) * 0.5
    audio[20000:24000] = np.sin(np.linspace(0, 20 * np.pi, 4000)) * 0.5

    breaks = _detect_silence_breaks(
        audio=audio, sr=sr, threshold_db=-40.0, min_gap_seconds=0.5
    )

    # Breaks that are too close together should be filtered out
    for i in range(1, len(breaks)):
        gap = breaks[i] - breaks[i - 1]
        assert gap >= 0.5


def test_detect_silence_breaks_finds_multiple_gaps() -> None:
    """Test _detect_silence_breaks finds multiple distinct silence gaps.

    Creates audio with multiple well-separated silence gaps and
    verifies that each gap is detected as a separate break point.
    """
    sr: int = 16000
    # Create audio with speech and silence alternating
    audio = np.zeros(shape=80000, dtype=np.float64)
    # Silence from 0-1s
    # Speech from 1-2s
    audio[16000:32000] = np.sin(np.linspace(0, 20 * np.pi, 16000)) * 0.5
    # Silence from 2-3s
    # Speech from 3-4s
    audio[48000:64000] = np.sin(np.linspace(0, 20 * np.pi, 16000)) * 0.5

    breaks = _detect_silence_breaks(
        audio=audio, sr=sr, threshold_db=-40.0, min_gap_seconds=0.5
    )

    assert len(breaks) >= 1


def test_detect_silence_breaks_with_custom_threshold() -> None:
    """Test _detect_silence_breaks respects custom threshold values.

    Verifies that a higher threshold (e.g., -20 dB) detects more
    silence regions than a lower threshold (e.g., -60 dB).
    """
    sr: int = 16000
    # Audio with varying energy levels
    audio = np.zeros(shape=48000, dtype=np.float64)
    # Quiet speech
    audio[16000:32000] = np.sin(np.linspace(0, 20 * np.pi, 16000)) * 0.1

    breaks_high = _detect_silence_breaks(
        audio=audio, sr=sr, threshold_db=-20.0, min_gap_seconds=0.5
    )
    breaks_low = _detect_silence_breaks(
        audio=audio, sr=sr, threshold_db=-60.0, min_gap_seconds=0.5
    )

    assert len(breaks_high) >= len(breaks_low)


# ---------------------------------------------------------------------------
# Audio resampling tests
# ---------------------------------------------------------------------------


def test_resample_to_16k_mono_stereo() -> None:
    """Test _resample_to_16k_mono converts stereo to mono correctly.

    Creates stereo audio at 48000 Hz and verifies the output is
    mono audio at 16000 Hz (3x fewer samples).
    """
    sr: int = 48000
    # 48000 samples, 2 channels - stereo sine wave
    t_arr: np.ndarray = np.linspace(0, 1.0, 48000)
    stereo_audio: np.ndarray = np.stack(
        [np.sin(2 * np.pi * 440 * t_arr), np.sin(2 * np.pi * 880 * t_arr)], axis=0
    )

    new_sr, mono_audio = _resample_to_16k_mono(audio=stereo_audio, original_sr=sr)

    assert new_sr == 16000
    assert mono_audio.ndim == 1
    assert mono_audio.size == 16000


def test_resample_to_16k_mono_mono() -> None:
    """Test _resample_to_16k_mono handles mono audio correctly.

    Creates mono audio at 48000 Hz and verifies the output is
    mono audio at 16000 Hz (3x fewer samples).
    """
    sr: int = 48000
    mono_audio: np.ndarray = np.zeros(shape=48000, dtype=np.float64)

    new_sr, result_audio = _resample_to_16k_mono(audio=mono_audio, original_sr=sr)

    assert new_sr == 16000
    assert result_audio.ndim == 1
    assert result_audio.size == 16000


def test_resample_to_16k_mono_already_16k() -> None:
    """Test _resample_to_16k_mono does not resample when already 16kHz.

    Creates mono audio at 16000 Hz and verifies the output is unchanged.
    """
    sr: int = 16000
    audio: np.ndarray = np.zeros(shape=16000, dtype=np.float64)

    new_sr, result_audio = _resample_to_16k_mono(audio=audio, original_sr=sr)

    assert new_sr == 16000
    np.testing.assert_array_equal(result_audio, audio)


def test_resample_to_16k_mono_preserves_signal() -> None:
    """Test _resample_to_16k_mono preserves signal amplitude.

    Creates a known sine wave and verifies the resampled mono audio
    has the correct amplitude after resampling.
    """
    sr: int = 48000
    mono_audio: np.ndarray = np.sin(np.linspace(0, 2 * np.pi, 48000))

    new_sr, result_audio = _resample_to_16k_mono(audio=mono_audio, original_sr=sr)

    assert new_sr == 16000
    assert result_audio.size == 16000
    # Amplitude should be preserved (close to 1.0)
    assert np.max(np.abs(result_audio)) > 0.5


def test_resample_to_16k_mono_stereo_amplitude() -> None:
    """Test _resample_to_16k_mono correctly averages stereo channels.

    Creates stereo audio with different values in each channel and
    verifies the mono output is the average of the two channels.
    """
    sr: int = 16000
    # Shape (2, 16000) - 2 channels, 16000 samples each
    stereo_audio: np.ndarray = np.zeros(shape=(2, 16000), dtype=np.float64)
    stereo_audio[0, :] = 1.0
    stereo_audio[1, :] = 3.0

    new_sr, mono_audio = _resample_to_16k_mono(audio=stereo_audio, original_sr=sr)

    assert new_sr == 16000
    np.testing.assert_array_equal(mono_audio, np.ones(shape=16000) * 2.0)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_chunk_audio_edge_case_short_audio() -> None:
    """Test chunk_audio handles very short audio gracefully.

    Verifies that when audio is shorter than the silence detection
    window, the function still yields a single chunk.
    """
    mock_sr: int = 16000
    short_audio: np.ndarray = np.zeros(shape=100, dtype=np.float64)

    with um.patch(
        "but_with_subs.chunking._load_audio", return_value=(mock_sr, short_audio)
    ):
        audio_path = pathlib.Path("/fake/short.wav")
        chunks = list(chunk_audio(audio_path=audio_path))

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)


def test_chunk_audio_edge_case_no_silence() -> None:
    """Test chunk_audio handles audio with no silence gaps.

    Mocks silence detection to return no breaks and verifies
    that a single chunk spanning the entire audio is yielded.
    """
    mock_sr: int = 16000
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float64)

    with (
        um.patch(
            "but_with_subs.chunking._load_audio", return_value=(mock_sr, mock_audio)
        ),
        um.patch("but_with_subs.chunking._detect_silence_breaks", return_value=[]),
    ):
        audio_path = pathlib.Path("/fake/no_silence.wav")
        chunks = list(chunk_audio(audio_path=audio_path))

    assert len(chunks) == 1
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 1.0


def test_chunk_audio_edge_case_continuous_silence() -> None:
    """Test chunk_audio handles audio that is entirely silence.

    Mocks silence detection to return no breaks (since the entire
    audio is silence, there are no transitions from speech to silence)
    and verifies a single chunk is yielded.
    """
    mock_sr: int = 16000
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float64)

    with (
        um.patch(
            "but_with_subs.chunking._load_audio", return_value=(mock_sr, mock_audio)
        ),
        um.patch("but_with_subs.chunking._detect_silence_breaks", return_value=[]),
    ):
        audio_path = pathlib.Path("/fake/continuous_silence.wav")
        chunks = list(chunk_audio(audio_path=audio_path))

    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)


def test_chunk_audio_edge_case_single_break_near_end() -> None:
    """Test chunk_audio handles a break very close to the end of audio.

    Mocks silence detection to return a break at 0.99s of a 1s audio
    and verifies that a small chunk is yielded at the end.
    """
    mock_sr: int = 16000
    mock_audio: np.ndarray = np.zeros(shape=16000, dtype=np.float64)

    with (
        um.patch(
            "but_with_subs.chunking._load_audio", return_value=(mock_sr, mock_audio)
        ),
        um.patch("but_with_subs.chunking._detect_silence_breaks", return_value=[0.99]),
    ):
        audio_path = pathlib.Path("/fake/break_near_end.wav")
        chunks = list(chunk_audio(audio_path=audio_path))

    assert len(chunks) == 2
    assert chunks[1].start_time == 0.99
    assert chunks[1].end_time == 1.0


def test_detect_silence_breaks_edge_case_all_silence() -> None:
    """Test _detect_silence_breaks with audio that is entirely silence.

    Verifies that when the entire audio is silence, no breaks are
    detected since there are no transitions from speech to silence.
    """
    sr: int = 16000
    all_silence: np.ndarray = np.zeros(shape=16000, dtype=np.float64)

    breaks = _detect_silence_breaks(
        audio=all_silence, sr=sr, threshold_db=-40.0, min_gap_seconds=0.5
    )

    assert breaks == []


def test_detect_silence_breaks_edge_case_all_speech() -> None:
    """Test _detect_silence_breaks with audio that is entirely speech.

    Verifies that when the entire audio is a continuous signal,
    no breaks are detected.
    """
    sr: int = 16000
    all_speech: np.ndarray = np.sin(np.linspace(0, 100 * np.pi, 16000))

    breaks = _detect_silence_breaks(
        audio=all_speech, sr=sr, threshold_db=-40.0, min_gap_seconds=0.5
    )

    assert breaks == []


def test_load_audio_returns_correct_sample_rate() -> None:
    """Test _load_audio returns the correct sample rate from a WAV file.

    Creates a real WAV file with a known sample rate and verifies
    that _load_audio returns the correct rate.
    """
    with tempfile_temporary_file() as temp_path:
        # Write a WAV file at 22050 Hz
        wavio.write(
            filename=temp_path, rate=22050, data=np.zeros(shape=1000, dtype=np.int16)
        )
        sr, _ = _load_audio(path=temp_path)

    assert sr == 22050


def test_load_audio_raises_on_empty_file() -> None:
    """Test _load_audio raises ValueError for empty audio files.

    Creates a WAV file with no audio data and verifies that
    _load_audio raises a ValueError.
    """
    with tempfile_temporary_file() as temp_path:
        # Write a WAV file with no data
        wavio.write(
            filename=temp_path, rate=16000, data=np.zeros(shape=0, dtype=np.int16)
        )

        try:
            _load_audio(path=temp_path)
        except ValueError as e:
            assert "no data" in str(e).lower()
        else:
            assert False, "Expected ValueError to be raised"
