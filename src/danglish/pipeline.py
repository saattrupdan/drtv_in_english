"""End-to-end processing pipeline as a streaming generator.

Yields :class:`ProgressEvent` objects so the FastAPI endpoint can forward
them to the client as a streaming response.
"""

import collections.abc as c

import openai

from .data_models import Chunk, DownloadProgress, ProgressEvent, VideoWithSubs
from .downloading import download
from .llm import correct_and_translate
from .logging_config import logger
from .vtt import parse_external_vtt, write_vtt_file

# Stage boundaries on the 0-100 progress scale.
DOWNLOAD_END = 50.0
TRANSLATE_END = 100.0


def run_pipeline(
    *,
    url: str,
    language: str,
    llm_client: openai.OpenAI,
    llm_model: str,
    max_parallel: int = 20,
) -> c.Iterator[ProgressEvent]:
    """Process ``url`` through download → translate → subtitle.

    Yields progress events on a 0-100 scale and finishes with a ``completed``
    event carrying the final :class:`VideoWithSubs` payload.

    Args:
        url: DR TV URL (episode, series, or film page).
        language: ISO-639-1 target language code (e.g. ``"en"``).
        llm_client: Pre-built OpenAI client for correct-and-translate.
        llm_model: Model name for the LLM API call.
        max_parallel: Maximum number of LLM requests in flight at once.

    Yields:
        :class:`ProgressEvent` objects describing pipeline progress.
    """
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

    if file.subtitles_path is None:
        yield ProgressEvent(
            stage="error",
            percentage=DOWNLOAD_END,
            message="No Danish subtitles available for this video",
        )
        return

    chunks: list[Chunk] = parse_external_vtt(path=file.subtitles_path)
    logger.info(f"Parsed {len(chunks)} cues from source subtitles")

    yield ProgressEvent(
        stage="translating",
        percentage=DOWNLOAD_END,
        message=f"Translating to {language}…",
    )

    llm_events: list[ProgressEvent] = []

    def _on_progress(ratio: float) -> None:
        pct = DOWNLOAD_END + ratio * (TRANSLATE_END - DOWNLOAD_END)
        llm_events.append(
            ProgressEvent(
                stage="translating",
                percentage=pct,
                message=f"Translating to {language}…",
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

    subtitles_path = file.video_path.with_suffix(f".{language}.vtt")
    write_vtt_file(chunks=chunks, path=subtitles_path)
    logger.info(f"Wrote {len(chunks)} cues to {subtitles_path}")

    yield ProgressEvent(
        stage="completed",
        percentage=TRANSLATE_END,
        message="Ready to watch!",
        result=VideoWithSubs(
            video_path=str(file.video_path.resolve()),
            subtitles_path=str(subtitles_path.resolve()),
        ),
    )
