"""FastAPI application for processing DR video URLs into subtitled videos."""

import collections.abc as c
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import openai
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .constants import DATA_DIR
from .data_models import ProgressEvent
from .llm import build_client
from .logging_config import configure_logging, logger
from .pipeline import run_pipeline


@dataclass
class AppState:
    """Container for objects shared across requests, attached to ``app.state``."""

    llm_client: openai.OpenAI
    llm_model: str


class ProcessRequest(BaseModel):
    """Body of a ``POST /process`` call."""

    url: str
    language: str = "en"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> c.AsyncIterator[None]:
    """Build the LLM client once at startup."""
    configure_logging()
    logger.info("Building LLM client")
    llm_client = build_client()
    llm_model = os.environ["LLM_MODEL"]
    app.state.app_state = AppState(llm_client=llm_client, llm_model=llm_model)
    logger.info("API ready")
    yield


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return service health.

    Returns:
        Mapping with a single ``status`` key set to ``"ok"``.
    """
    return {"status": "ok"}


@router.post("/process")
def process(req: ProcessRequest, request: Request) -> StreamingResponse:
    """Run the full processing pipeline as a streaming NDJSON response.

    Each line is a JSON-encoded :class:`ProgressEvent`. The final line has
    ``stage="completed"`` and includes the ``result`` payload.

    Returns:
        ``StreamingResponse`` yielding newline-delimited JSON.
    """
    state: AppState = request.app.state.app_state

    def stream() -> c.Iterator[bytes]:
        try:
            for event in run_pipeline(
                url=req.url,
                language=req.language,
                llm_client=state.llm_client,
                llm_model=state.llm_model,
            ):
                if event.result is not None:
                    event.result.video_path = _to_media_url(event.result.video_path)
                    event.result.subtitles_path = _to_media_url(
                        event.result.subtitles_path
                    )
                yield _encode(event)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline failed")
            yield _encode(
                ProgressEvent(stage="error", percentage=0.0, message=str(exc))
            )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _to_media_url(path: str) -> str:
    """Convert an absolute filesystem path under ``DATA_DIR`` to a browser URL.

    Returns:
        URL of the form ``/api/media/<filename>`` for the static mount.
    """
    return f"/api/media/{Path(path).name}"


def _encode(event: ProgressEvent) -> bytes:
    r"""Serialise an event as a single NDJSON line.

    Returns:
        UTF-8 encoded JSON line terminated with ``\n``.
    """
    return (json.dumps(event.model_dump(mode="json")) + "\n").encode("utf-8")


app = FastAPI(title="danglish", lifespan=lifespan)
app.include_router(router)

_data_dir = Path(DATA_DIR)
_data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(_data_dir)), name="media")
