"""Loading audio for automatic speech recognition."""

from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.signal
import torch

import torchaudio

from .constants import TARGET_SAMPLE_RATE
from .logging_config import logger


def _trim_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -40.0, min_silence_secs: float = 0.3) -> np.ndarray:
    """Remove leading and trailing silence.

    Uses RMS energy in 20 ms frames. Frames below threshold_db are
    considered silent.  min_silence_secs controls how much silence
    is kept at each end (to avoid chopping words).
    """
    frame_size = int(0.02 * sample_rate)  # 20 ms frames
    hop_size = frame_size // 2
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(frame_size), mode='valid') / frame_size)
    rms = rms[::hop_size]  # hop-aligned sub-samples
    threshold_linear = 10 ** (threshold_db / 20.0)
    non_silent = np.where(rms > threshold_linear)[0]

    if len(non_silent) == 0:
        return audio  # entirely silent, return unchanged

    # Keep min_silence_secs of buffer at each end
    buffer_frames = int(min_silence_secs / 0.01)  # 10 ms frames
    start = max(0, non_silent[0] - buffer_frames)
    end = min(len(audio), non_silent[-1] + buffer_frames + 1)
    return audio[start:end]


def _normalize_loudness(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize AND RMS-normalize audio to a target loudness.

    wav2vec2 was trained on audio with approximately -16 dBFS RMS.
    """
    # Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    # RMS normalization to approx -16 dBFS (~0.14 linear)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 0.14
        audio = audio * (target_rms / rms)

    # Re-clamp to prevent clipping after RMS scaling
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak

    return audio


def _high_pass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """Apply a high-pass Butterworth filter to remove low-frequency rumble."""
    nyquist = sample_rate / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    b, a = scipy.signal.butter(N=2, Wn=normalized_cutoff, btype='high')
    # filtfilt gives zero-phase filtering
    return scipy.signal.filtfilt(b, a, audio)


def load_audio(path: Path) -> np.ndarray:
    """Load a WAV file and return preprocessed mono audio at 16kHz.

    Applies high-pass filtering, silence trimming, RMS loudness
    normalization, and resampling to 16 kHz.  Converts multi-channel
    audio to mono.

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

    # NEW: High-pass filter to remove low-frequency rumble
    audio_data = _high_pass_filter(audio_data, sample_rate)

    # NEW: Trim leading and trailing silence
    audio_data = _trim_silence(audio_data, sample_rate)

    # NEW: Loudness normalization (peak + RMS)
    audio_data = _normalize_loudness(audio_data)

    # NEW: Resample using torchaudio (sinc-based interpolation)
    audio_data = _resample_to_16k_mono(audio=audio_data, original_sr=sample_rate)

    logger.info(f"Loaded audio from {path} at {sample_rate:,} Hz")
    return audio_data


def _resample_to_16k_mono(audio: np.ndarray, original_sr: int) -> np.ndarray:
    """Resample audio to 16 kHz using torchaudio's sinc-based resampler.

    torchaudio.functional.resample uses sinc interpolation which is
    superior to scipy.signal.resample (FFT-based) for ASR-quality audio.

    Args:
        audio:
            Mono audio float data array, of shape (audio_len,).
        original_sr:
            The original sample rate of the audio.

    Returns:
        Resampled audio array at 16 kHz.

    """
    if original_sr == TARGET_SAMPLE_RATE:
        return audio

    # torchaudio expects shape (channels, samples)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
    resampled = torchaudio.functional.resample(
        audio_tensor,
        orig_freq=int(original_sr),
        new_freq=TARGET_SAMPLE_RATE,
    )
    return resampled.squeeze(0).numpy()


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
