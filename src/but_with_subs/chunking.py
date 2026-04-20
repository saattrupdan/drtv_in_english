"""Audio chunking functionality for splitting audio into segments.

This module provides functions to split audio files into chunks based on
natural breaks detected through silence detection.
"""

import logging
import pathlib as pl
from collections.abc import Generator

import numpy as np
import scipy.io.wavfile
import scipy.signal
from pydantic import BaseModel

logger = logging.getLogger(__package__)


class Chunk(BaseModel):
    """A chunk of audio data.

    Attributes:
        start_time:
            Start time of the chunk, in seconds from the beginning of the audio.
        end_time:
            End time of the chunk, in seconds from the beginning of the audio.
        audio:
            Mono audio data of the chunk, as a numpy array in 16 kHz.

    Config:
        arbitrary_types:
            Allow numpy arrays as field types.
    """

    model_config = {"arbitrary_types_allowed": True}

    start_time: float
    end_time: float
    audio: np.ndarray


def chunk_audio(audio_path: pl.Path) -> Generator[Chunk, None, None]:
    """Yield audio chunks split by silence breaks.

    Loads the audio file, resamples it to 16 kHz mono, detects silence
    breaks, and yields Chunk models for each segment between breaks.

    Args:
        audio_path:
            Path to the WAV audio file to split.

    Yields:
        Chunk models for each audio segment between silence breaks.

    """
    sample_rate, audio = _load_audio(path=audio_path)
    sr, mono_audio = _resample_to_16k_mono(audio=audio, original_sr=sample_rate)
    break_times = _detect_silence_breaks(
        audio=mono_audio, sr=sr, threshold_db=-40.0, min_gap_seconds=0.5
    )
    for start_time, end_time, chunk_data in _split_audio_into_chunks(
        audio=mono_audio, break_times=break_times, start_time=0.0
    ):
        yield Chunk(start_time=start_time, end_time=end_time, audio=chunk_data)


def _load_audio(path: pl.Path) -> tuple[int, np.ndarray]:
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
    sample_rate, audio_data = scipy.io.wavfile.read(filename=path)
    if audio_data.size == 0:
        raise ValueError(f"Audio file {path} contains no data")

    # Ensure that the audio array is a numpy array of floats
    audio_data = np.array(audio_data, dtype=np.float32) / np.iinfo(audio_data.dtype).max

    # Ensure that the audio is mono
    if audio_data.ndim > 1:
        audio_data = np.mean(a=audio_data, axis=1)

    logger.info("Loaded audio from %s at %d Hz", path, sample_rate)
    return sample_rate, audio_data


def _resample_to_16k_mono(
    audio: np.ndarray, original_sr: int
) -> tuple[int, np.ndarray]:
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
    target_sr = 16000
    if target_sr != original_sr:
        logger.info("Resampling audio from %d Hz to 16,000 Hz", original_sr)
        n_samples = int(audio.size * target_sr / original_sr)
        mono_audio = scipy.signal.resample(x=audio, num=n_samples)
    logger.info("Resampled audio from %d Hz to %d Hz", original_sr, target_sr)
    return target_sr, mono_audio


def _detect_silence_breaks(
    audio: np.ndarray, sr: int, threshold_db: float, min_gap_seconds: float
) -> list[float]:
    """Detect silence breaks in audio using energy-based threshold.

    Finds gaps in the audio where the signal energy falls below the
    specified threshold. Only gaps longer than min_gap_seconds are
    returned as break points.

    Args:
        audio:
            Mono audio data array.
        sr:
            Sample rate of the audio.
        threshold_db:
            Energy threshold in dB below which a silence gap is detected.
        min_gap_seconds:
            Minimum duration of silence in seconds to be considered a break.

    Returns:
        List of break times in seconds from the start of the audio.

    """
    target_sr = 16000
    window_size = int(target_sr * 0.05)
    hop_size = int(target_sr * 0.025)

    n_windows = (len(audio) - window_size) // hop_size + 1
    if n_windows <= 0:
        logger.warning("Audio too short for silence detection")
        return []

    energies = np.zeros(shape=n_windows)
    for i in range(n_windows):
        start = i * hop_size
        end = start + window_size
        frame = audio[start:end]
        energies[i] = 10.0 * np.log10(np.max(np.abs(frame) ** 2) + 1e-10)

    silence_mask = energies < threshold_db
    break_times: list[float] = []

    for i in range(1, n_windows):
        if silence_mask[i] and (not silence_mask[i - 1]):
            break_time = float(i * hop_size) / sr
            break_times.append(break_time)

    if len(break_times) > 1:
        filtered: list[float] = []
        for i in range(1, len(break_times)):
            gap = break_times[i] - break_times[i - 1]
            if gap >= min_gap_seconds:
                filtered.append(break_times[i])
        break_times = filtered

    logger.info("Detected %d silence breaks", len(break_times))
    return break_times


def _split_audio_into_chunks(
    audio: np.ndarray, break_times: list[float], start_time: float
) -> list[tuple[float, float, np.ndarray]]:
    """Split audio into chunks based on silence break times.

    Divides the audio into segments using the provided break times.
    Each chunk is represented as a tuple of (start_time, end_time, data).

    Args:
        audio:
            Mono audio data array.
        break_times:
            List of break times in seconds from the start of the audio.
        start_time:
            Start time offset for the current audio segment.

    Returns:
        List of tuples containing (start_time, end_time, audio_data)
        for each chunk.

    """
    sr = 16000
    chunks: list[tuple[float, float, np.ndarray]] = []

    times = [start_time] + break_times + [float(len(audio)) / sr]

    for i in range(len(times) - 1):
        chunk_start = times[i]
        chunk_end = times[i + 1]
        start_sample = int(chunk_start * sr)
        end_sample = int(chunk_end * sr)
        chunk_data = audio[start_sample:end_sample]
        if chunk_data.size > 0:
            chunks.append((chunk_start, chunk_end, chunk_data))

    logger.info("Split audio into %d chunks", len(chunks))
    return chunks
