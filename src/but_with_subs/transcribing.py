"""Audio transcription using Hugging Face automatic speech recognition.

This module provides functions to transcribe audio into word-level text segments
using a pretrained ASR pipeline from the Hugging Face transformers library.
"""

import collections.abc as c
import typing as t

import bits_and_bobs as bnb
import numpy as np
import torch
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.pipeline import Pipeline
from tqdm.auto import tqdm
from transformers import AutomaticSpeechRecognitionPipeline

from .constants import MIN_CHUNK_LENGTH_SECONDS, TARGET_SAMPLE_RATE
from .data_models import Chunk
from .device import get_device
from .logging_config import logger


def _merge_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping or adjacent segments.

    Args:
        segments: List of (start, end) time tuples.

    Returns:
        Merged non-overlapping segments.
    """
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: s[0])
    merged: list[list[float]] = [list(sorted_segs[0])]
    for start, end in sorted_segs[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s[0], s[1]) for s in merged]


def vad_segment_audio(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    segment_duration: float = 10.0,
    overlap: float = 2.0,
) -> list[tuple[float, float, np.ndarray]]:
    """Split audio into VAD-segmented pieces using pyannote VAD.

    Returns list of (start_time, end_time, audio_slice) tuples.
    Only non-silent segments are returned.

    Args:
        audio: Mono audio array at sample_rate.
        sample_rate: Audio sample rate (default: 16000).
        segment_duration: Max duration of each VAD segment in seconds.
        overlap: Overlap between consecutive segments in seconds.

    Returns:
        List of (start, end, audio_slice) tuples for non-silent regions.
    """
    vad: Pipeline = VoiceActivityDetection().instantiate(
        {
            "onset": 0.5,
            "offset": 0.3,
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        }
    )
    vad.to(get_device())

    waveform = torch.from_numpy(audio).unsqueeze(dim=0).float()
    with ProgressHook() as hook:
        speech = vad.apply(
            file={"waveform": waveform, "sample_rate": sample_rate}, hook=hook
        )

    active_segments = [(seg.start, seg.end) for seg in speech.itersegments()]

    if not active_segments:
        return [(0.0, len(audio) / sample_rate, audio)]  # fallback: entire audio

    # Merge overlapping active segments and split into fixed-duration windows
    merged = _merge_segments(active_segments)
    pieces: list[tuple[float, float, np.ndarray]] = []
    for seg_start, seg_end in merged:
        seg_duration = seg_end - seg_start
        offset = 0.0
        while offset < seg_duration:
            chunk_end = min(offset + segment_duration, seg_duration)
            if chunk_end - offset < 1.0:  # skip tiny leftovers
                break
            audio_start = int((seg_start + offset) * sample_rate)
            audio_end = int((seg_start + chunk_end) * sample_rate)
            pieces.append(
                (
                    seg_start + offset,
                    seg_start + chunk_end,
                    audio[audio_start:audio_end],
                )
            )
            offset += segment_duration - overlap  # overlap for continuity

    return pieces


def transcribe_audio(
    audio: np.ndarray,
    model: AutomaticSpeechRecognitionPipeline,
    min_chunk_length: float = MIN_CHUNK_LENGTH_SECONDS,
    show_progress: bool = True,
    on_progress: c.Callable[[float], None] | None = None,
) -> list[Chunk]:
    """Transcribe audio with VAD pre-segmentation.

    Uses voice activity detection to split audio into shorter segments
    before passing to the ASR pipeline.  This prevents context drift on
    long audio where the transformer's self-attention over the full
    sequence degrades word-level timestamp accuracy.

    Args:
        audio:
            Full audio array (mono, float, 16 kHz).
        model:
            The ASR pipeline to use for transcription.
        min_chunk_length (optional):
            Minimum segment duration in seconds. Segments shorter than this
            are excluded. Defaults to ``MIN_CHUNK_LENGTH_SECONDS`` (0.05 s).
        show_progress (optional):
            Whether to display a progress bar. Defaults to ``True``.
        on_progress (optional):
            Callback invoked after each VAD segment finishes, receiving a
            float in ``[0, 1]`` indicating completion ratio.

    Returns:
        List of word-level transcribed ``Chunk`` objects.
    """
    # VAD pre-segmentation
    vad_segments = vad_segment_audio(audio)

    word_chunks: list[Chunk] = list()
    total_steps = len(vad_segments)

    with tqdm(
        total=total_steps, desc="Transcribing", disable=not show_progress
    ) as pbar:
        for seg_start, seg_end, seg_audio in vad_segments:
            if seg_audio.size == 0:
                pbar.update(1)
                continue
            try:
                with bnb.no_terminal_output():
                    result = t.cast(dict, model(seg_audio, return_timestamps="word"))
            except Exception as e:
                logger.error(
                    f"Transcription failed for segment "
                    f"[{seg_start:.2f}-{seg_end:.2f}]: {e}"
                )
                pbar.update(1)
                continue

            for transcription_dct in result["chunks"]:
                local_start = float(transcription_dct["timestamp"][0])
                local_end = float(transcription_dct["timestamp"][1])

                # Adjust timestamps to global audio coordinates
                global_start = seg_start + local_start
                global_end = seg_start + local_end

                if global_end - global_start < min_chunk_length:
                    continue

                # Extract audio for this word segment (global indices)
                audio_start = int(TARGET_SAMPLE_RATE * global_start)
                audio_end = int(TARGET_SAMPLE_RATE * global_end)
                audio_end = min(audio_end, len(audio))
                segment_audio = audio[audio_start:audio_end]

                word_chunks.append(
                    Chunk(
                        start_time=global_start,
                        end_time=global_end,
                        audio=segment_audio,
                        text=transcription_dct["text"],
                        speaker=None,
                    )
                )
            pbar.update(1)
            if on_progress is not None and total_steps > 0:
                on_progress(pbar.n / total_steps)

    logger.info(f"Completed transcription of {len(word_chunks)} word segments")

    return word_chunks


def assign_speakers(
    word_chunks: list[Chunk],
    turns: list[tuple[float, float, str]],
) -> list[Chunk]:
    """Assign speakers to word-level chunks based on temporal overlap.

    Each chunk is assigned the speaker whose turn most strongly overlaps
    the chunk's time range. If multiple turns overlap, the one with the
    largest intersection is chosen.

    Args:
        word_chunks:
            Word-level ``Chunk`` objects produced by :func:`transcribe_audio`.
        turns:
            Diarisation output from :func:`diarize`, i.e. a list of
            ``(start, end, speaker)`` tuples.

    Returns:
            The same ``Chunk`` objects with ``.speaker`` populated.
    """
    if not turns:
        return word_chunks

    for chunk in word_chunks:
        best_speaker: str | None = None
        best_overlap: float = 0.0

        for turn_start, turn_end, speaker in turns:
            # Compute overlap between chunk and turn
            overlap_start = max(chunk.start_time, turn_start)
            overlap_end = min(chunk.end_time, turn_end)
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_speaker = speaker

        chunk.speaker = best_speaker

    return word_chunks
