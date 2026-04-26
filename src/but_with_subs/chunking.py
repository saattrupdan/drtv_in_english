"""Audio chunking functionality for splitting audio into segments.

This module provides functions to split audio files into chunks based on
natural breaks detected through silence detection.
"""

import logging
import pathlib as pl

import numpy as np
import scipy.io.wavfile
import scipy.signal
import torch
from pydantic import BaseModel
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm.auto import tqdm

logger = logging.getLogger(__package__)


class Chunk(BaseModel):
    """A chunk of audio data.

    Attributes:
        start_time:
            Start time of the chunk in seconds.
        end_time:
            End time of the chunk in seconds.
        audio:
            Numpy array containing the audio data for the chunk.
    """

    model_config = {"arbitrary_types_allowed": True}

    start_time: float
    end_time: float
    audio: np.ndarray


def chunk_audio(audio_path: pl.Path) -> list[Chunk]:
    """Split audio into chunks based on silence breaks.

    Loads the audio file, resamples it to 16 kHz mono, detects silence
    breaks, and returns Chunk models for each segment between breaks.

    Args:
        audio_path:
            Path to the audio file to chunk.

    Returns:
        List of Chunk models for each audio segment between silence breaks.

    """
    sample_rate, audio = _load_audio(path=audio_path)
    mono_audio = _resample_to_16k_mono(audio=audio, original_sr=sample_rate)
    return _split_audio_into_chunks(audio=mono_audio)


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

    logger.info(f"Loaded audio from {path} at {sample_rate:,} Hz")
    return sample_rate, audio_data


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
    target_sr = 16000
    if target_sr != original_sr:
        logger.info(f"Resampling audio from {original_sr:,} Hz to 16,000 Hz...")
        n_samples = int(audio.size * target_sr / original_sr)
        mono_audio = scipy.signal.resample(x=audio, num=n_samples)
    logger.info(f"Resampled audio from {original_sr:,} Hz to {target_sr:,} Hz")
    return mono_audio


def _split_audio_into_chunks(audio: np.ndarray) -> list[Chunk]:
    """Split audio into chunks.

    Args:
        audio:
            Mono audio data array.

    Returns:
        List of Chunk models for each segment.
    """
    with tqdm(total=100, desc="Splitting audio into chunks", unit="chunk") as pbar:
        speech_timestamps = get_speech_timestamps(
            audio=torch.from_numpy(audio),
            model=load_silero_vad(),
            return_seconds=True,
            threshold=0.4,
            progress_tracking_callback=lambda progress: pbar.update(
                int(progress) - pbar.n
            ),
        )
    chunks = []
    for speech_timestamp_dct in speech_timestamps:
        start_s = speech_timestamp_dct["start"]
        end_s = speech_timestamp_dct["end"]
        chunk_audio = audio[int(start_s * 16_000) : int(end_s * 16_000)]
        chunk = Chunk(start_time=start_s, end_time=end_s, audio=chunk_audio)
        chunks.append(chunk)

    logger.info(f"Split audio into {len(chunks)} chunks")
    return chunks
