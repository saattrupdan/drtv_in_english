"""Tests for the audio_loading module."""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import scipy.io.wavfile

from but_with_subs.audio_loading import (
    _resample_to_16k_mono,
    load_audio,
    validate_audio,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_wav_file(tmp_path: Path) -> Path:
    """Create a temporary valid WAV file.

    Returns:
        A Path to the created temporary WAV file.
    """
    sample_rate = 16_000
    duration_seconds = 1.0
    n_samples = int(sample_rate * duration_seconds)
    audio_data = np.sin(
        2 * np.pi * 440 * np.linspace(0, duration_seconds, n_samples)
    ).astype(np.int16)
    file_path = tmp_path / "test_audio.wav"
    scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=audio_data)
    return file_path


@pytest.fixture
def temp_mp3_file(tmp_path: Path) -> Path:
    """Create a temporary MP3 file (simulated as WAV for testing).

    Returns:
        A Path to the created temporary MP3 file.
    """
    # Note: True MP3 files require pydub or similar libraries
    # For testing purposes, we'll create a WAV file but test the handling
    sample_rate = 16_000
    duration_seconds = 0.5
    n_samples = int(sample_rate * duration_seconds)
    audio_data = np.sin(
        2 * np.pi * 440 * np.linspace(0, duration_seconds, n_samples)
    ).astype(np.int16)
    file_path = tmp_path / "test_audio.mp3"
    scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=audio_data)
    return file_path


@pytest.fixture
def stereo_audio_file(tmp_path: Path) -> Path:
    """Create a temporary stereo WAV file.

    Returns:
        A Path to the created temporary stereo WAV file.
    """
    sample_rate = 44_100
    duration_seconds = 1.0
    n_samples = int(sample_rate * duration_seconds)
    left_channel = np.sin(
        2 * np.pi * 440 * np.linspace(0, duration_seconds, n_samples)
    ).astype(np.int16)
    right_channel = np.sin(
        2 * np.pi * 880 * np.linspace(0, duration_seconds, n_samples)
    ).astype(np.int16)
    stereo_data = np.column_stack((left_channel, right_channel))
    file_path = tmp_path / "stereo_audio.wav"
    scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=stereo_data)
    return file_path


@pytest.fixture
def empty_wav_file(tmp_path: Path) -> Path:
    """Create an empty WAV file.

    Returns:
        A Path to the created empty WAV file.
    """
    file_path = tmp_path / "empty_audio.wav"
    file_path.touch()
    return file_path


@pytest.fixture
def corrupted_wav_file(tmp_path: Path) -> Path:
    """Create a corrupted WAV file.

    Returns:
        A Path to the created corrupted WAV file.
    """
    file_path = tmp_path / "corrupted_audio.wav"
    file_path.write_bytes(b"not a valid wav file content" + b"\x00" * 100)
    return file_path


@pytest.fixture
def very_short_audio_file(tmp_path: Path) -> Path:
    """Create a very short audio file (few samples).

    Returns:
        A Path to the created very short audio file.
    """
    sample_rate = 16_000
    audio_data = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.int16)
    file_path = tmp_path / "very_short.wav"
    scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=audio_data)
    return file_path


@pytest.fixture
def high_sample_rate_audio_file(tmp_path: Path) -> Path:
    """Create an audio file with high sample rate.

    Returns:
        A Path to the created high sample rate audio file.
    """
    sample_rate = 48_000
    duration_seconds = 0.5
    n_samples = int(sample_rate * duration_seconds)
    audio_data = np.sin(
        2 * np.pi * 440 * np.linspace(0, duration_seconds, n_samples)
    ).astype(np.int16)
    file_path = tmp_path / "high_sample_rate.wav"
    scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=audio_data)
    return file_path


# =============================================================================
# Tests for validate_audio() function
# =============================================================================


class TestValidateAudio:
    """Tests for the validate_audio() function."""

    def test_valid_audio_at_correct_sample_rate(self) -> None:
        """Test validation of valid audio at the correct sample rate."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16_000)).astype(np.float32)
        result = validate_audio(audio=audio, sample_rate=16_000)

        assert result["is_valid"] is True
        assert result["sample_rate_valid"] is True
        assert result["duration_valid"] is True
        assert result["has_signal"] is True
        assert len(result["errors"]) == 0

    def test_invalid_audio_wrong_sample_rate(self) -> None:
        """Test validation rejects audio with wrong sample rate."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44_100)).astype(np.float32)
        result = validate_audio(audio=audio, sample_rate=44_100)

        assert result["is_valid"] is False
        assert result["sample_rate_valid"] is False
        assert any("does not match expected" in err for err in result["errors"])

    def test_invalid_audio_too_short(self) -> None:
        """Test validation rejects audio that is too short."""
        audio = np.array([0.5, -0.5], dtype=np.float32)
        result = validate_audio(
            audio=audio, sample_rate=16_000, min_duration_seconds=1.0
        )

        assert result["is_valid"] is False
        assert result["duration_valid"] is False
        assert any("below minimum" in err for err in result["errors"])

    def test_invalid_audio_too_long(self) -> None:
        """Test validation rejects audio that is too long."""
        # Create 2 hours of audio at 16kHz
        audio = np.zeros(16_000 * 3600 * 2, dtype=np.float32)
        result = validate_audio(
            audio=audio, sample_rate=16_000, max_duration_seconds=3600.0
        )

        assert result["is_valid"] is False
        assert result["duration_valid"] is False
        assert any("exceeds maximum" in err for err in result["errors"])

    def test_silent_audio_rejected(self) -> None:
        """Test validation rejects silent audio."""
        audio = np.zeros(16_000, dtype=np.float32)
        result = validate_audio(audio=audio, sample_rate=16_000)

        assert result["is_valid"] is False
        assert result["has_signal"] is False
        assert any("silent" in err for err in result["errors"])

    def test_audio_with_extreme_amplitude(self) -> None:
        """Test validation of audio with amplitude exceeding normalized range."""
        audio = np.full(16_000, 2.0, dtype=np.float64)
        result = validate_audio(audio=audio, sample_rate=16_000)

        assert result["is_valid"] is False
        assert any("exceeds normalized range" in err for err in result["errors"])

    def test_audio_with_wrong_dtype(self) -> None:
        """Test validation of audio with non-float dtype."""
        audio = np.zeros(16_000, dtype=np.int16)
        result = validate_audio(audio=audio, sample_rate=16_000)

        assert result["is_valid"] is False
        assert any("not a float type" in err for err in result["errors"])

    def test_returns_duration_seconds(self) -> None:
        """Test that validation returns correct duration."""
        audio = np.zeros(32_000, dtype=np.float32)
        result = validate_audio(audio=audio, sample_rate=16_000)

        assert result["duration_seconds"] == 2.0

    def test_returns_peak_amplitude(self) -> None:
        """Test that validation returns correct peak amplitude."""
        audio = np.array([0.0, 0.5, -0.8, 0.3], dtype=np.float32)
        result = validate_audio(audio=audio, sample_rate=16_000)

        assert np.isclose(result["peak_amplitude"], 0.8)

    def test_raises_error_for_empty_array(self) -> None:
        """Test that empty audio array raises ValueError."""
        audio = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="Audio array cannot be empty"):
            validate_audio(audio=audio, sample_rate=16_000)

    def test_raises_error_for_non_numpy_array(self) -> None:
        """Test that non-numpy array raises ValueError."""
        audio = [0.1, 0.2, 0.3]

        with pytest.raises(ValueError, match="Audio must be a numpy array"):
            validate_audio(audio=audio, sample_rate=16_000)  # type: ignore

    def test_raises_error_for_invalid_sample_rate(self) -> None:
        """Test that invalid sample rate raises TypeError."""
        audio = np.zeros(1000, dtype=np.float32)

        with pytest.raises(TypeError, match="Sample rate must be a positive integer"):
            validate_audio(audio=audio, sample_rate="16000")  # type: ignore

    def test_raises_error_for_negative_sample_rate(self) -> None:
        """Test that negative sample rate raises TypeError."""
        audio = np.zeros(1000, dtype=np.float32)

        with pytest.raises(TypeError, match="Sample rate must be a positive integer"):
            validate_audio(audio=audio, sample_rate=-1)


class TestValidateAudioEdgeCases:
    """Edge case tests for validate_audio()."""

    def test_audio_at_minimum_duration(self) -> None:
        """Test validation of audio at the minimum duration boundary."""
        audio = np.zeros(int(16_000 * 0.1), dtype=np.float32)
        audio[0] = 0.5  # Add signal to avoid silence error
        result = validate_audio(
            audio=audio, sample_rate=16_000, min_duration_seconds=0.1
        )

        assert result["is_valid"] is True

    def test_audio_at_maximum_duration(self) -> None:
        """Test validation of audio at the maximum duration boundary."""
        audio = np.zeros(int(16_000 * 3600), dtype=np.float32)
        audio[0] = 0.5  # Add signal to avoid silence error
        result = validate_audio(
            audio=audio, sample_rate=16_000, max_duration_seconds=3600.0
        )

        assert result["is_valid"] is True

    def test_audio_just_above_threshold(self) -> None:
        """Test validation of audio just above silence threshold."""
        audio = np.full(16_000, 0.0011, dtype=np.float32)
        result = validate_audio(audio=audio, sample_rate=16_000)

        assert result["is_valid"] is True
        assert result["has_signal"] is True

    def test_audio_just_below_threshold(self) -> None:
        """Test validation of audio just below silence threshold."""
        audio = np.full(16_000, 0.0009, dtype=np.float32)
        result = validate_audio(audio=audio, sample_rate=16_000)

        assert result["is_valid"] is False
        assert result["has_signal"] is False

    def test_custom_sample_rate(self) -> None:
        """Test validation with custom expected sample rate."""
        audio = np.zeros(44_100, dtype=np.float32)
        audio[0] = 0.5
        result = validate_audio(
            audio=audio, sample_rate=44_100, expected_sample_rate=44_100
        )

        assert result["is_valid"] is True


# =============================================================================
# Tests for load_audio() function
# =============================================================================


class TestLoadAudioValidFiles:
    """Tests for loading valid audio files."""

    def test_load_valid_wav_file(self, temp_wav_file: Path) -> None:
        """Test loading a valid WAV file."""
        audio = load_audio(path=temp_wav_file)

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.ndim == 1  # Should be mono
        assert len(audio) > 0

    def test_load_audio_returns_normalized_data(self, temp_wav_file: Path) -> None:
        """Test that loaded audio is normalized to float range [-1, 1]."""
        audio = load_audio(path=temp_wav_file)

        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)

    def test_load_stereo_audio_converts_to_mono(self, stereo_audio_file: Path) -> None:
        """Test that stereo audio is converted to mono."""
        audio = load_audio(path=stereo_audio_file)

        assert audio.ndim == 1, "Stereo audio should be converted to mono"

    def test_load_audio_at_different_sample_rates(
        self, high_sample_rate_audio_file: Path
    ) -> None:
        """Test loading audio with different sample rates."""
        audio = load_audio(path=high_sample_rate_audio_file)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0


class TestLoadAudioFileFormats:
    """Tests for handling different file formats."""

    def test_load_wav_format(self, temp_wav_file: Path) -> None:
        """Test loading WAV format specifically."""
        audio = load_audio(path=temp_wav_file)

        assert audio is not None
        assert len(audio) > 0

    def test_load_mp3_format_simulation(self, temp_mp3_file: Path) -> None:
        """Test loading MP3 format (simulated as WAV for testing)."""
        # Note: Real MP3 support would require additional libraries
        # This test verifies the file path handling
        audio = load_audio(path=temp_mp3_file)

        assert audio is not None
        assert len(audio) > 0


class TestLoadAudioErrorHandling:
    """Tests for error handling in load_audio()."""

    def test_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Test that non-existent files raise ValueError."""
        non_existent_path = tmp_path / "non_existent.wav"

        with pytest.raises(ValueError, match="Audio file not found"):
            load_audio(path=non_existent_path)

    def test_invalid_audio_file_raises_error(self, corrupted_wav_file: Path) -> None:
        """Test that corrupted/invalid audio files raise ValueError."""
        with pytest.raises(
            ValueError, match="Failed to read audio file|Error reading audio file"
        ):
            load_audio(path=corrupted_wav_file)

    def test_empty_file_raises_error(self, empty_wav_file: Path) -> None:
        """Test that empty files raise ValueError."""
        with pytest.raises(
            ValueError, match="Failed to read audio file|Error reading audio file"
        ):
            load_audio(path=empty_wav_file)

    def test_permission_error_handling(self, tmp_path: Path) -> None:
        """Test handling of permission errors."""
        file_path = tmp_path / "no_permission.wav"
        sample_rate = 16_000
        audio_data = np.zeros(1000, dtype=np.int16)
        scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=audio_data)

        # Make file unreadable (this may not work on all systems)
        file_path.chmod(0o000)

        try:
            with pytest.raises(
                ValueError, match="Failed to read audio file|Error reading audio file"
            ):
                load_audio(path=file_path)
        finally:
            # Restore permissions for cleanup
            file_path.chmod(0o644)


class TestLoadAudioSampleRateValidation:
    """Tests for sample rate handling and validation."""

    def test_resampling_to_16k(self, high_sample_rate_audio_file: Path) -> None:
        """Test that audio is resampled to 16kHz."""
        audio = load_audio(path=high_sample_rate_audio_file)

        # After loading, audio should be at 16kHz
        # We can verify by checking the length is appropriate for the duration
        assert len(audio) > 0

    def test_16k_audio_no_resampling_needed(self, temp_wav_file: Path) -> None:
        """Test that 16kHz audio doesn't require resampling."""
        audio = load_audio(path=temp_wav_file)

        assert audio is not None
        assert len(audio) > 0


class TestLoadAudioChannelHandling:
    """Tests for mono/stereo channel handling."""

    def test_mono_audio_preserved(self, temp_wav_file: Path) -> None:
        """Test that mono audio is handled correctly."""
        audio = load_audio(path=temp_wav_file)

        assert audio.ndim == 1

    def test_stereo_audio_converted_to_mono(self, stereo_audio_file: Path) -> None:
        """Test that stereo audio is properly converted to mono."""
        audio = load_audio(path=stereo_audio_file)

        assert audio.ndim == 1
        assert len(audio) > 0


# =============================================================================
# Tests for _resample_to_16k_mono() function
# =============================================================================


class TestResampleTo16kMono:
    """Tests for the _resample_to_16k_mono helper function."""

    def test_resample_higher_sample_rate(self) -> None:
        """Test resampling from higher sample rate to 16kHz."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48_000)).astype(np.float32)
        result = _resample_to_16k_mono(audio=audio, original_sr=48_000)

        assert isinstance(result, np.ndarray)
        assert len(result) == int(16_000)

    def test_resample_lower_sample_rate(self) -> None:
        """Test resampling from lower sample rate to 16kHz."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 8_000)).astype(np.float32)
        result = _resample_to_16k_mono(audio=audio, original_sr=8_000)

        assert isinstance(result, np.ndarray)
        assert len(result) == int(16_000)

    def test_no_resampling_when_already_16k(self) -> None:
        """Test that no resampling occurs when already at 16kHz."""
        original_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16_000)).astype(
            np.float32
        )
        result = _resample_to_16k_mono(audio=original_audio, original_sr=16_000)

        # When no resampling is needed, the function returns the original audio
        np.testing.assert_allclose(original_audio, result, rtol=1e-5)

    def test_resample_preserves_signal_characteristics(self) -> None:
        """Test that resampling preserves the basic signal characteristics."""
        original_sr = 44_100
        duration = 0.5
        n_samples = int(original_sr * duration)
        frequency = 440  # Hz
        audio = np.sin(
            2 * np.pi * frequency * np.linspace(0, duration, n_samples)
        ).astype(np.float32)

        result = _resample_to_16k_mono(audio=audio, original_sr=original_sr)

        assert len(result) > 0
        assert np.all(np.abs(result) <= 1.5)  # Allow some overshoot from resampling


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in audio loading."""

    def test_very_short_audio_file(self, very_short_audio_file: Path) -> None:
        """Test loading very short audio files."""
        audio = load_audio(path=very_short_audio_file)

        assert audio is not None
        assert len(audio) > 0

    def test_very_long_audio_file(self, tmp_path: Path) -> None:
        """Test loading very long audio files."""
        sample_rate = 16_000
        duration_seconds = 60  # 1 minute
        n_samples = sample_rate * duration_seconds
        audio_data = np.sin(
            2 * np.pi * 440 * np.linspace(0, duration_seconds, n_samples)
        ).astype(np.int16)
        long_file = tmp_path / "long_audio.wav"
        scipy.io.wavfile.write(filename=long_file, rate=sample_rate, data=audio_data)

        audio = load_audio(path=long_file)

        assert audio is not None
        assert len(audio) == int(16_000 * duration_seconds)

    def test_corrupted_audio_file(self, corrupted_wav_file: Path) -> None:
        """Test handling of corrupted audio files."""
        with pytest.raises(ValueError):
            load_audio(path=corrupted_wav_file)

    def test_audio_with_zero_values(self, tmp_path: Path) -> None:
        """Test loading audio with all zero values."""
        sample_rate = 16_000
        audio_data = np.zeros(16_000, dtype=np.int16)
        file_path = tmp_path / "zero_audio.wav"
        scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=audio_data)

        audio = load_audio(path=file_path)

        assert audio is not None
        assert np.all(audio == 0.0)

    def test_audio_with_extreme_values(self, tmp_path: Path) -> None:
        """Test loading audio with extreme (max/min) amplitude values."""
        sample_rate = 16_000
        duration = 1.0
        # Use a sine wave at max amplitude (realistic audio with extreme values)
        t = np.linspace(0, duration, sample_rate, endpoint=False)
        audio_data = (np.sin(2 * np.pi * 440 * t) * np.iinfo(np.int16).max).astype(
            np.int16
        )
        file_path = tmp_path / "extreme_audio.wav"
        scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=audio_data)

        audio = load_audio(path=file_path)

        assert audio is not None
        # After peak normalization + RMS normalization to -16 dBFS,
        # a sine wave should have values within [-1, 1] and RMS near 0.14
        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)
        rms = float(np.sqrt(np.mean(audio**2)))
        assert np.isclose(rms, 0.14, atol=0.02)

    def test_multichannel_audio_handling(self, tmp_path: Path) -> None:
        """Test handling of multi-channel (e.g., 5.1) audio."""
        sample_rate = 48_000
        duration_seconds = 0.5
        n_samples = int(sample_rate * duration_seconds)
        # Create 6-channel audio (simulating 5.1)
        audio_data = np.random.randint(
            -32768, 32767, size=(n_samples, 6), dtype=np.int16
        )
        file_path = tmp_path / "multichannel_audio.wav"
        scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=audio_data)

        audio = load_audio(path=file_path)

        assert audio is not None
        assert audio.ndim == 1  # Should be converted to mono

    def test_logging_on_successful_load(
        self, temp_wav_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that successful audio loading is logged."""
        with caplog.at_level(logging.INFO):
            load_audio(path=temp_wav_file)

        assert any("Loaded audio from" in record.message for record in caplog.records)

    def test_logging_on_resampling(
        self, high_sample_rate_audio_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that resampling is logged."""
        with caplog.at_level(logging.INFO):
            load_audio(path=high_sample_rate_audio_file)

        assert any("Resampling audio" in record.message for record in caplog.records)


# =============================================================================
# Mocking Tests
# =============================================================================


class TestWithMocking:
    """Tests using mocking for specific scenarios."""

    def test_scipy_read_error_handling(self, temp_wav_file: Path) -> None:
        """Test error handling when scipy.io.wavfile.read raises an exception."""
        with patch("scipy.io.wavfile.read") as mock_read:
            mock_read.side_effect = OSError("Simulated I/O error")

            with pytest.raises(ValueError, match="Failed to read audio file"):
                load_audio(path=temp_wav_file)

    def test_generic_exception_handling(self, temp_wav_file: Path) -> None:
        """Test handling of generic exceptions from scipy."""
        with patch("scipy.io.wavfile.read") as mock_read:
            mock_read.side_effect = Exception("Unexpected error")

            with pytest.raises(ValueError, match="Error reading audio file"):
                load_audio(path=temp_wav_file)

    def test_load_audio_no_data_raises_value_error(self, temp_wav_file: Path) -> None:
        """Test that load_audio raises ValueError when file contains no data.

        This tests line 39 in audio_loading.py where ValueError is raised
        when scipy.io.wavfile.read returns an empty array.
        """
        with patch("scipy.io.wavfile.read") as mock_read:
            # Mock scipy.io.wavfile.read to return an empty array
            mock_read.return_value = (16000, np.array([], dtype=np.int16))

            with pytest.raises(ValueError, match="Audio file.*contains no data"):
                load_audio(path=temp_wav_file)


# =============================================================================
# Integration Tests
# =============================================================================


class TestAudioLoadingIntegration:
    """Integration tests for the audio loading module."""

    def test_full_audio_loading_pipeline(self, tmp_path: Path) -> None:
        """Test the complete audio loading pipeline with various parameters."""
        # Create audio with specific characteristics
        sample_rate = 44_100
        frequency = 440  # Hz
        duration = 2.0
        n_samples = int(sample_rate * duration)

        # Create a sine wave
        t = np.linspace(0, duration, n_samples)
        audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        file_path = tmp_path / "sine_wave.wav"
        scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=audio_data)

        # Load the audio
        loaded_audio = load_audio(path=file_path)

        # Verify the loaded audio
        assert loaded_audio is not None
        assert loaded_audio.dtype == np.float32
        assert loaded_audio.ndim == 1
        # Note: resampling can introduce slight overshoot due to Gibbs phenomenon
        assert np.all(np.abs(loaded_audio) <= 1.5)

        # Verify resampling: 2 seconds at 44.1kHz should become 2 seconds at 16kHz
        expected_length = int(16_000 * duration)
        assert len(loaded_audio) == expected_length

    def test_audio_quality_after_resampling(self, tmp_path: Path) -> None:
        """Test that audio quality is preserved after resampling."""
        sample_rate = 44_100
        frequency = 440  # Hz
        duration = 1.0
        n_samples = int(sample_rate * duration)

        # Create a pure tone
        t = np.linspace(0, duration, n_samples)
        original_audio = np.sin(2 * np.pi * frequency * t)
        audio_data = (original_audio * 32767).astype(np.int16)

        file_path = tmp_path / "pure_tone.wav"
        scipy.io.wavfile.write(filename=file_path, rate=sample_rate, data=audio_data)

        loaded_audio = load_audio(path=file_path)

        # The resampled audio should still have the same frequency characteristics
        # We can check this by looking at the zero-crossings
        zero_crossings = np.sum(np.diff(np.sign(loaded_audio)) != 0)
        # For a 440Hz tone at 16kHz for 1 second, we expect
        # approximately 880 zero-crossings
        expected_zero_crossings = int(2 * frequency * duration)
        # Allow some tolerance due to resampling
        assert (
            abs(zero_crossings - expected_zero_crossings)
            < expected_zero_crossings * 0.1
        )
