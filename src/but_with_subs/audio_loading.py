"""Loading audio for automatic speech recording."""

import logging
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.signal

from .constants import TARGET_SAMPLE_RATE

logger = logging.getLogger(__package__)


def load_audio(path: Path) -> np.ndarray:
    """Load a WAV file and return normalized mono audio at 16kHz.

    Resamples to 16kHz if needed and converts multi-channel to mono.

    Args:
        path: Path to the WAV file.

    Returns:
        Float numpy array of mono audio data.

    Raises:
        ValueError: If file cannot be read or contains no audio data.
    """
    try:
        sample_rate, audio_data = scipy.io.wavfile.read(filename=path)
    except FileNotFoundError:
        raise ValueError(f"Audio file not found: {path}") from None
    except OSError as e:
        raise ValueError(f"Failed to read audio file {path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error reading audio file {path}: {e}") from e

    if audio_data.size == 0:
        raise ValueError(f"Audio file {path} contains no data")

    # Ensure that the audio array is a numpy array of floats
    audio_data = np.array(audio_data, dtype=np.float32) / np.iinfo(audio_data.dtype).max

    # Ensure that the audio is mono
    if audio_data.ndim > 1:
        audio_data = np.mean(a=audio_data, axis=1)

    # Downsample to 16 kHz
    audio_data = _resample_to_16k_mono(audio=audio_data, original_sr=sample_rate)

    logger.info(f"Loaded audio from {path} at {sample_rate:,} Hz")
    return audio_data


def _resample_to_16k_mono(audio: np.ndarray, original_sr: int) -> np.ndarray:
    """Resample audio to 16 kHz mono using scipy.signal.resample.

    Converts multi-channel audio to mono by averaging channels and
    resamples the audio to 16 kHz if the original sample rate differs.

    Args:
        audio:
            Mono audio float data array, of shape (audio_len,).
        original_sr:
            The original sample rate of the audio.

    Returns:
        Resampled audio array at 16 kHz.

    """
    target_sr = TARGET_SAMPLE_RATE
    if target_sr != original_sr:
        logger.info(f"Resampling audio from {original_sr:,} Hz to 16,000 Hz...")
        n_samples = int(audio.size * target_sr / original_sr)
        return scipy.signal.resample(x=audio, num=n_samples)
    return audio


def validate_audio(
    audio: np.ndarray,
    sample_rate: int,
    expected_sample_rate: int = TARGET_SAMPLE_RATE,
    min_duration_seconds: float = 0.1,
    max_duration_seconds: float = 3600.0,
) -> dict[str, bool | str | float | list[str]]:
    """Validate audio data for speech recognition.

    Checks audio for common issues including:
    - Sample rate correctness
    - Duration within acceptable bounds
    - Audio amplitude range
    - Presence of signal (not silence)
    - Data type and format

    Args:
        audio:
            Audio data as a numpy array.
        sample_rate:
            Sample rate of the audio in Hz.
        expected_sample_rate:
            Expected sample rate (default: 16000 Hz).
        min_duration_seconds:
            Minimum acceptable duration in seconds.
        max_duration_seconds:
            Maximum acceptable duration in seconds.

    Returns:
        Dictionary with validation results:
            - 'is_valid': bool - Whether audio passes all checks
            - 'sample_rate_valid': bool - Whether sample rate is correct
            - 'duration_valid': bool - Whether duration is within bounds
            - 'has_signal': bool - Whether audio contains non-silent data
            - 'duration_seconds': float - Duration of the audio
            - 'peak_amplitude': float - Maximum absolute amplitude
            - 'errors': list[str] - List of validation error messages

    Raises:
        ValueError: If audio is not a numpy array or is empty.
        TypeError: If sample_rate is not an integer.
    """
    if not isinstance(audio, np.ndarray):
        raise ValueError("Audio must be a numpy array")

    if audio.size == 0:
        raise ValueError("Audio array cannot be empty")

    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise TypeError("Sample rate must be a positive integer")

    errors: list[str] = []

    # Check sample rate
    sample_rate_valid = sample_rate == expected_sample_rate
    if not sample_rate_valid:
        errors.append(
            f"Sample rate {sample_rate} Hz does not match "
            f"expected {expected_sample_rate} Hz"
        )

    # Calculate duration
    duration_seconds = audio.size / sample_rate
    duration_valid = min_duration_seconds <= duration_seconds <= max_duration_seconds
    if not duration_valid:
        if duration_seconds < min_duration_seconds:
            errors.append(
                f"Audio duration {duration_seconds:.3f}s is below "
                f"minimum {min_duration_seconds}s"
            )
        if duration_seconds > max_duration_seconds:
            errors.append(
                f"Audio duration {duration_seconds:.1f}s exceeds "
                f"maximum {max_duration_seconds}s"
            )

    # Check for signal presence (not silence)
    peak_amplitude = float(np.max(np.abs(audio)))
    has_signal = peak_amplitude > 0.001  # Threshold for silence

    if not has_signal:
        errors.append("Audio appears to be silent (peak amplitude below threshold)")

    # Check amplitude range
    amplitude_valid = peak_amplitude <= 1.0
    if not amplitude_valid:
        errors.append(
            f"Audio amplitude {peak_amplitude:.3f} exceeds normalized range [-1, 1]"
        )

    # Check data type
    dtype_valid = audio.dtype in (np.float32, np.float64)
    if not dtype_valid:
        errors.append(f"Audio data type {audio.dtype} is not a float type")

    is_valid = (
        sample_rate_valid
        and duration_valid
        and has_signal
        and amplitude_valid
        and dtype_valid
    )

    return {
        "is_valid": is_valid,
        "sample_rate_valid": sample_rate_valid,
        "duration_valid": duration_valid,
        "has_signal": has_signal,
        "duration_seconds": duration_seconds,
        "peak_amplitude": peak_amplitude,
        "errors": errors,
    }
