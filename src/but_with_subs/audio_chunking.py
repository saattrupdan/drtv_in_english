"""Audio chunking functionality for splitting audio into segments."""

import logging
import os
import pathlib as pl
from typing import cast

import numpy as np
import scipy.io.wavfile
import scipy.signal
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from .data_models import Chunk

logger = logging.getLogger(__package__)


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
    return mono_audio


def _split_audio_into_chunks(audio: np.ndarray) -> list[Chunk]:
    """Split audio into chunks.

    Args:
        audio:
            Mono audio data array.

    Returns:
        List of Chunk models for each segment.
    """
    pipeline = cast(
        Pipeline,
        Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=os.getenv("HF_TOKEN")
        ),
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    pipeline.to(device)

    with ProgressHook() as hook:
        speech_timestamps = pipeline(
            dict(waveform=torch.from_numpy(audio).unsqueeze(dim=0), sample_rate=16_000),
            hook=hook,
        ).speaker_diarization

    chunks = []
    for turn, speaker in speech_timestamps:
        start_s = turn.start
        end_s = turn.end
        if end_s - start_s < 0.05:  # Required for Wav2Vec2 models
            continue
        chunk_audio = audio[int(start_s * 16_000) : int(end_s * 16_000)]
        chunk = Chunk(
            start_time=start_s, end_time=end_s, audio=chunk_audio, speaker=speaker
        )
        chunks.append(chunk)

    logger.info(f"Split audio into {len(chunks)} chunks")
    return chunks
