"""Loading audio for automatic speech recognition."""

import logging
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.signal

logger = logging.getLogger(__package__)


def load_audio(path: Path) -> np.ndarray:
    """Load a WAV audio file using scipy.io.wavfile.

    Reads the specified WAV file and returns the sample rate and audio
    data array.

    Args:
        path:
            Path to the WAV file to load.

    Returns:
        Tuple of (sample_rate, audio_data).

    Raises:
        ValueError:
            If the file cannot be read or contains no audio data.
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
        Tuple of (new_sample_rate, mono_resampled_audio).

    """
    target_sr = 16_000
    if target_sr != original_sr:
        logger.info(f"Resampling audio from {original_sr:,} Hz to 16,000 Hz...")
        n_samples = int(audio.size * target_sr / original_sr)
        mono_audio = scipy.signal.resample(x=audio, num=n_samples)
    return mono_audio
