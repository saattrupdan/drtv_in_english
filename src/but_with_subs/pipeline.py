"""End-to-end processing pipeline as a streaming generator.

Yields :class:`ProgressEvent` objects so the FastAPI endpoint can forward
them to the client as a streaming response. Heavy ML models are passed in
rather than constructed here, so the API can load them once at startup.
"""

import collections.abc as c
from pathlib import Path

import numpy as np
import openai
from punctfix.inference import PunctFixer
from pyannote.audio import Pipeline
from sqlalchemy.engine import Engine
from sqlmodel import Session
from transformers import AutomaticSpeechRecognitionPipeline

from .audio_extraction import extract_audio
from .audio_loading import load_audio
from .constants import MAX_WORDS
from .data_models import Chunk, DownloadProgress, ProgressEvent, VideoWithSubs
from .database import upsert_file
from .downloading import download
from .llm import correct_and_translate
from .logging_config import logger
from .subtitling import generate_subtitles
from .text_chunking import group_word_chunks
from .transcribing import assign_speakers, transcribe_audio
from .vtt import parse_external_vtt

# Stage boundaries on the 0-100 progress scale.
DOWNLOAD_END = 50.0
TRANSCRIBE_END = 95.0
SUBTITLE_END = 100.0


def _diarize(audio: np.ndarray, model: Pipeline) -> list[tuple[float, float, str]]:
    """Run diarisation on audio using the given Pipeline.

    Args:
        audio: Mono audio array at ``TARGET_SAMPLE_RATE``.
        model: A pre-loaded diarisation Pipeline.

    Returns:
        List of ``(start, end, speaker)`` tuples.
    """
    from .audio_chunking import diarize

    return diarize(audio, model)


def run_pipeline(
    *,
    url: str,
    language: str,
    asr_model: AutomaticSpeechRecognitionPipeline,
    punctuation_model: PunctFixer,
    llm_client: openai.OpenAI,
    llm_model: str,
    engine: Engine,
    diarization_model: Pipeline | None = None,
    max_parallel: int = 20,
) -> c.Iterator[ProgressEvent]:
    """Process ``url`` through download → transcribe → translate → subtitle.

    Yields progress events on a 0-100 scale and finishes with a ``completed``
    event carrying the final :class:`VideoWithSubs` payload.

    Args:
        url: Source media URL.
        language: ISO-639-1 code; transcripts are always translated.
        asr_model: Pre-loaded ASR pipeline.
        punctuation_model: Pre-loaded punctuation model.
        llm_client: Pre-built OpenAI client for correct-and-translate.
        llm_model: Model name for the LLM API call.
        engine: SQLModel engine for persisting the file record.
        diarization_model: Optional pre-loaded diarisation Pipeline.
        max_parallel: Maximum number of LLM requests in flight at once.

    Yields:
        :class:`ProgressEvent` objects describing pipeline progress.
    """
    # --- Download ---------------------------------------------------------
    yield ProgressEvent(
        stage="downloading", percentage=0.0, message="Starting download…"
    )

    latest_download_pct = {"value": 0.0}

    def _on_download(progress: DownloadProgress) -> None:
        latest_download_pct["value"] = max(
            latest_download_pct["value"], progress.percentage
        )

    file = download(url=url, progress_hook=_on_download)
    yield ProgressEvent(
        stage="downloading", percentage=DOWNLOAD_END, message="Download complete"
    )

    if file.video_path is None:
        yield ProgressEvent(
            stage="error", percentage=DOWNLOAD_END, message="Download produced no video"
        )
        return

    # If the source provided real subtitles, skip the entire ASR pipeline:
    # those cues are already segmented and free of transcription errors, so
    # we go straight to translation. Otherwise fall back to extracting audio
    # and running ASR + diarisation + punctuation-based grouping.
    if file.subtitles_path is not None:
        with Session(engine) as session:
            upsert_file(
                session=session,
                url=url,
                video_path=file.video_path.resolve(),
            )
        yield ProgressEvent(
            stage="transcribing",
            percentage=DOWNLOAD_END,
            message="Using source subtitles…",
        )
        chunks: list[Chunk] = parse_external_vtt(path=file.subtitles_path)
        logger.info(f"Parsed {len(chunks)} cues from source subtitles")
        output_vtt_dir = file.video_path
    else:
        # Persist the file record now that we have video/audio paths.
        audio_path = extract_audio(video_path=file.video_path)
        with Session(engine) as session:
            upsert_file(
                session=session,
                url=url,
                video_path=file.video_path.resolve(),
                audio_path=audio_path.resolve(),
            )

        yield ProgressEvent(
            stage="transcribing", percentage=DOWNLOAD_END, message="Loading audio…"
        )
        audio = load_audio(path=audio_path)

        transcribe_events: list[ProgressEvent] = []

        def _on_transcribe(ratio: float) -> None:
            pct = DOWNLOAD_END + ratio * (TRANSCRIBE_END - DOWNLOAD_END)
            transcribe_events.append(
                ProgressEvent(
                    stage="transcribing", percentage=pct, message="Transcribing audio…"
                )
            )

        word_chunks = transcribe_audio(
            audio=audio,
            model=asr_model,
            show_progress=False,
            on_progress=_on_transcribe,
        )
        yield from transcribe_events

        if diarization_model is not None:
            yield ProgressEvent(
                stage="transcribing",
                percentage=TRANSCRIBE_END,
                message="Running speaker diarisation…",
            )
            turns = _diarize(audio, diarization_model)
            word_chunks = assign_speakers(word_chunks, turns)
            logger.info(
                f"Assigned speakers to "
                f"{sum(1 for c in word_chunks if c.speaker is not None)} "
                f"of {len(word_chunks)} chunks"
            )

        chunks = group_word_chunks(
            word_chunks=word_chunks,
            punctuation_model=punctuation_model,
            max_words=MAX_WORDS,
        )
        logger.info(f"Grouped into {len(chunks)} text segments")
        output_vtt_dir = audio_path

    # --- Translate -------------------------------------------------------
    yield ProgressEvent(
        stage="transcribing",
        percentage=TRANSCRIBE_END,
        message=f"Translating to {language}…",
    )

    llm_events: list[ProgressEvent] = []

    def _on_progress(ratio: float) -> None:
        pct = TRANSCRIBE_END + ratio * (SUBTITLE_END - TRANSCRIBE_END)
        llm_events.append(
            ProgressEvent(
                stage="transcribing",
                percentage=pct,
                message=f"Correcting + translating to {language}…",
            )
        )

    chunks = correct_and_translate(
        chunks,
        target_language=language,
        client=llm_client,
        model=llm_model,
        max_parallel=max_parallel,
        on_progress=_on_progress,
    )
    yield from llm_events

    # --- Subtitles --------------------------------------------------------
    yield ProgressEvent(
        stage="subtitling", percentage=TRANSCRIBE_END, message="Generating subtitles…"
    )
    subtitles_path = output_vtt_dir.with_suffix(".vtt")
    generate_subtitles(
        chunks=chunks, audio_path=output_vtt_dir, output_path=subtitles_path
    )

    with Session(engine) as session:
        upsert_file(session=session, url=url, subtitles_path=subtitles_path.resolve())

    # --- Done -------------------------------------------------------------
    yield ProgressEvent(
        stage="completed",
        percentage=SUBTITLE_END,
        message="Ready to watch!",
        result=VideoWithSubs(
            video_path=str(_resolve_path(file.video_path)),
            subtitles_path=str(_resolve_path(subtitles_path)),
        ),
    )


def _resolve_path(path: Path) -> Path:
    """Return the absolute path of ``path`` without raising if it does not exist."""
    return path.resolve() if path.exists() else path.absolute()
