"""FastAPI application for processing video URLs into subtitled videos."""

import collections.abc as c
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import bits_and_bobs as bnb
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from punctfix.inference import PunctFixer
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    M2M100ForConditionalGeneration,
    pipeline,
)

from .constants import ASR_MODEL_ID, DATA_DIR, TRANSLATION_MODEL
from .data_models import ProgressEvent
from .database import build_engine, init_db
from .device import get_device
from .logging_config import configure_logging, logger
from .pipeline import run_pipeline
from .tokenization_small100 import SMALL100Tokenizer


@dataclass
class AppState:
    """Container for objects shared across requests, attached to ``app.state``."""

    engine: Engine
    asr_model: AutomaticSpeechRecognitionPipeline
    punctuation_model: PunctFixer
    translation_model: M2M100ForConditionalGeneration
    translation_tokenizer: SMALL100Tokenizer


class ProcessRequest(BaseModel):
    """Body of a ``POST /process`` call."""

    url: str
    language: str | None = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> c.AsyncIterator[None]:
    """Load models + initialise the database once at startup."""
    configure_logging()

    engine = build_engine()
    init_db(engine)

    device = get_device()
    logger.info(f"Loading ASR model: {ASR_MODEL_ID}")
    with bnb.no_terminal_output():
        asr_model = pipeline(
            task="automatic-speech-recognition",
            model=ASR_MODEL_ID,
            device=device,
            num_beams=5,
        )

    logger.info("Loading punctuation model")
    with bnb.no_terminal_output():
        punctuation_model = PunctFixer(language="da")

    logger.info(f"Loading translation model: {TRANSLATION_MODEL}")
    with bnb.no_terminal_output():
        translation_model = M2M100ForConditionalGeneration.from_pretrained(
            TRANSLATION_MODEL
        )
        translation_tokenizer = SMALL100Tokenizer.from_pretrained(
            TRANSLATION_MODEL, tgt_lang="en"
        )
        translation_model = translation_model.to(device)  # ty: ignore[invalid-argument-type]

    app.state.app_state = AppState(
        engine=engine,
        asr_model=asr_model,
        punctuation_model=punctuation_model,
        translation_model=translation_model,
        translation_tokenizer=translation_tokenizer,
    )
    logger.info("API ready")
    try:
        yield
    finally:
        engine.dispose()


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

    if req.language:
        state.translation_tokenizer.set_tgt_lang_special_tokens(req.language)

    def stream() -> c.Iterator[bytes]:
        try:
            for event in run_pipeline(
                url=req.url,
                language=req.language,
                asr_model=state.asr_model,
                punctuation_model=state.punctuation_model,
                translation_model=state.translation_model,
                translation_tokenizer=state.translation_tokenizer,
                engine=state.engine,
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
        A URL of the form ``/api/media/<filename>`` that the proxy maps to
        the static media mount on this app.
    """
    return f"/api/media/{Path(path).name}"


def _encode(event: ProgressEvent) -> bytes:
    r"""Serialise an event as a single NDJSON line.

    Returns:
        UTF-8 encoded JSON line terminated with ``\n``.
    """
    return (json.dumps(event.model_dump(mode="json")) + "\n").encode("utf-8")


app = FastAPI(title="but_with_subs", lifespan=lifespan)
app.include_router(router)

_data_dir = Path(DATA_DIR)
_data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(_data_dir)), name="media")
