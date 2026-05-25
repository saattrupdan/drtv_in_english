"""FastAPI app: resolve a DR URL, proxy HLS, and stream translated cues.

Flow:

1. ``POST /prepare`` → resolve URL via yt-dlp (no download), fetch DR's
   Danish ``.vtt`` once, parse it, register a :class:`Job`, and kick
   off background translation. Responds immediately.
2. Browser plays ``GET /stream/{job}/master.m3u8`` via hls.js.
3. Browser opens ``GET /translate/{job}`` (NDJSON) and renders each
   translated cue in an overlay as it arrives.
"""

import collections.abc as c
import contextlib
import json
import os
import threading
from dataclasses import dataclass

import httpx
import openai
import yt_dlp
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .data_models import CueEvent, PrepareResponse
from .hls_proxy import HlsRegistry, fetch_and_rewrite, is_playlist
from .jobs import Job, JobRegistry
from .llm import build_client, correct_and_translate
from .logging_config import configure_logging, logger
from .resolver import resolve
from .vtt import parse_vtt_text


@dataclass
class AppState:
    """Container for objects shared across requests, attached to ``app.state``."""

    llm_client: openai.OpenAI
    llm_model: str
    jobs: JobRegistry
    http: httpx.AsyncClient
    sync_http: httpx.Client


class PrepareRequest(BaseModel):
    """Body of a ``POST /prepare`` call."""

    url: str
    language: str = "en"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> c.AsyncIterator[None]:
    """Build shared clients at startup; close them at shutdown."""
    configure_logging()
    logger.info("Building LLM client")
    llm_client = build_client()
    llm_model = os.environ["LLM_MODEL"]
    http = httpx.AsyncClient(follow_redirects=True, timeout=60.0)
    sync_http = httpx.Client(follow_redirects=True, timeout=60.0)
    app.state.app_state = AppState(
        llm_client=llm_client,
        llm_model=llm_model,
        jobs=JobRegistry(),
        http=http,
        sync_http=sync_http,
    )
    logger.info("API ready")
    try:
        yield
    finally:
        await http.aclose()
        sync_http.close()


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return service health.

    Returns:
        Mapping with a single ``status`` key set to ``"ok"``.
    """
    return {"status": "ok"}


DRTV_SEARCH_URL = "https://prod95-cdn.dr-massive.com/api/v2/search"
DRTV_SEARCH_PARAMS = {
    "device": "web_browser",
    "ff": "idp,ldp,rpt",
    "group": "true",
    "lang": "da",
    "segments": "drtv,optedout",
    "sub": "Registered",
    "userId": "8820df5e01fa4fd8bd6bd3e49b81950f",
}
DRTV_BASE = "https://www.dr.dk/drtv"


def _simplify_item(item: dict, kind: str) -> dict:
    """Reduce a DRTV item to the fields the frontend needs."""
    return {
        "title": item.get("title") or "",
        "subtitle": (item.get("customFields") or {}).get("ContextualTitleExtraDetails")
        or (item.get("customFields") or {}).get("ExtraDetails")
        or "",
        "description": item.get("shortDescription") or "",
        "image": (item.get("images") or {}).get("tile")
        or (item.get("images") or {}).get("wallpaper")
        or "",
        "path": item.get("path") or "",
        "url": f"{DRTV_BASE}{item.get('path') or ''}",
        "kind": kind,
    }


@router.get("/search")
async def search(q: str, request: Request) -> dict:
    """Search DRTV by free-text query.

    Returns:
        ``{"series": [...], "playable": [...]}`` — each result has a
        normalised set of fields the frontend can render directly.
    """
    q = q.strip()
    if not q:
        return {"series": [], "playable": []}

    state: AppState = request.app.state.app_state
    params = {**DRTV_SEARCH_PARAMS, "term": q}
    response = await state.http.get(DRTV_SEARCH_URL, params=params)
    if response.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"DRTV search failed ({response.status_code})"
        )
    data = response.json()
    series = [_simplify_item(it, "series") for it in (data.get("series") or {}).get("items") or []]
    playable = [_simplify_item(it, "playable") for it in (data.get("playable") or {}).get("items") or []]
    return {"series": series, "playable": playable}


@router.get("/episodes")
def episodes(path: str, request: Request) -> dict:
    """List episodes for a DRTV series path.

    Args:
        path:
            A series path such as ``/serie/badehoteller_404047``.

    Returns:
        ``{"title": str, "episodes": [...]}`` where each episode has
        ``title``, ``subtitle``, ``image``, and ``url`` fields ready to
        feed into ``POST /prepare``.
    """
    if not path.startswith("/serie/") and not path.startswith("/saeson/"):
        raise HTTPException(status_code=422, detail="Path must be a series or season")

    series_url = f"{DRTV_BASE}{path}"
    episodes_out: list[dict] = []
    series_title = ""

    try:
        opts = {"quiet": True, "extract_flat": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            top = ydl.extract_info(series_url, download=False, process=True)
            series_title = top.get("title") or ""

            queue: list[dict] = list(top.get("entries") or [])
            while queue:
                entry = queue.pop(0)
                entry_url = entry.get("url") or ""
                if "/episode/" in entry_url:
                    episodes_out.append(
                        {
                            "title": entry.get("title") or "",
                            "subtitle": entry.get("description") or "",
                            "image": entry.get("thumbnail") or "",
                            "url": entry_url,
                            "episode_number": entry.get("episode_number"),
                            "season_number": entry.get("season_number"),
                        }
                    )
                elif "/saeson/" in entry_url:
                    season = ydl.extract_info(entry_url, download=False, process=True)
                    queue.extend(season.get("entries") or [])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Failed to list episodes: {exc}"
        ) from exc

    return {"title": series_title, "episodes": episodes_out}


@router.post("/prepare")
def prepare(req: PrepareRequest, request: Request) -> PrepareResponse:
    """Resolve a DR URL, fetch its subs, and start translating in the background.

    Returns:
        :class:`PrepareResponse` with proxy URLs the browser can start
        using immediately. Translation continues in the background and
        finished cues are exposed via ``GET /translate/{job_id}``.

    Raises:
        HTTPException:
            4xx for resolution failures (no HLS / no Danish subs).
    """
    state: AppState = request.app.state.app_state

    try:
        media = resolve(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = state.sync_http.get(media.subtitle_url, headers=media.subtitle_headers)
    response.raise_for_status()
    chunks = parse_vtt_text(response.content.decode("utf-8", errors="replace"))
    if not chunks:
        raise HTTPException(status_code=422, detail="Source subtitles are empty")

    registry = HlsRegistry(headers=media.hls_headers)
    job = state.jobs.create(
        title=media.title,
        hls_master_url=media.hls_url,
        registry=registry,
        chunks=chunks,
    )

    threading.Thread(
        target=_translate_job,
        args=(job, req.language, state.llm_client, state.llm_model),
        daemon=True,
        name=f"translate-{job.job_id}",
    ).start()

    return PrepareResponse(
        job_id=job.job_id,
        title=media.title,
        hls_url=f"/api/stream/{job.job_id}/master.m3u8",
        cue_count=len(chunks),
    )


@router.get("/stream/{job_id}/master.m3u8")
async def get_master(job_id: str, request: Request) -> Response:
    """Fetch and rewrite the HLS master playlist for ``job_id``.

    Returns:
        Rewritten ``application/vnd.apple.mpegurl`` playlist.
    """
    job = _require_job(request, job_id)
    state: AppState = request.app.state.app_state
    body, ctype = await fetch_and_rewrite(
        client=state.http,
        url=job.hls_master_url,
        registry=job.registry,
        proxy_prefix=f"/api/stream/{job_id}/p/",
    )
    return Response(content=body, media_type=ctype)


@router.get("/stream/{job_id}/p/{token}")
async def proxy(job_id: str, token: str, request: Request) -> Response:
    """Proxy an upstream URL previously registered for ``job_id``.

    Playlists are rewritten on the fly; segments are streamed as-is
    with DR's required headers attached.

    Returns:
        Playlist :class:`Response` or :class:`StreamingResponse` for
        binary segments.

    Raises:
        HTTPException:
            404 if ``token`` is not registered for this job.
    """
    job = _require_job(request, job_id)
    upstream = job.registry.resolve(token)
    if upstream is None:
        raise HTTPException(status_code=404, detail="Unknown proxy token")
    state: AppState = request.app.state.app_state

    if is_playlist(upstream, None):
        body, ctype = await fetch_and_rewrite(
            client=state.http,
            url=upstream,
            registry=job.registry,
            proxy_prefix=f"/api/stream/{job_id}/p/",
        )
        return Response(content=body, media_type=ctype)

    headers = dict(job.registry.headers)
    range_header = request.headers.get("range")
    if range_header:
        headers["Range"] = range_header
    upstream_req = state.http.build_request("GET", upstream, headers=headers)
    upstream_resp = await state.http.send(upstream_req, stream=True)

    media_type = upstream_resp.headers.get("content-type", "application/octet-stream")
    response_headers: dict[str, str] = {}
    for hop in ("content-length", "content-range", "accept-ranges"):
        if hop in upstream_resp.headers:
            response_headers[hop] = upstream_resp.headers[hop]

    async def body() -> c.AsyncIterator[bytes]:
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(
        body(),
        status_code=upstream_resp.status_code,
        media_type=media_type,
        headers=response_headers,
    )


@router.get("/translate/{job_id}")
def stream_translation(job_id: str, request: Request) -> StreamingResponse:
    """Stream translated cues for ``job_id`` as NDJSON.

    Replays any cues already produced, then blocks for new ones until
    the translation thread signals completion or an error.

    Returns:
        ``application/x-ndjson`` stream of :class:`CueEvent` objects.
    """
    job = _require_job(request, job_id)

    def stream() -> c.Iterator[bytes]:
        cursor = 0
        while True:
            with job.condition:
                while cursor == len(job.cues) and not job.done:
                    job.condition.wait(timeout=30.0)
                pending = job.cues[cursor:]
                cursor = len(job.cues)
                finished = job.done
                error = job.error
            for cue in pending:
                yield (json.dumps(cue.model_dump()) + "\n").encode("utf-8")
            if finished:
                if error:
                    yield (json.dumps({"done": True, "error": error}) + "\n").encode(
                        "utf-8"
                    )
                else:
                    yield (json.dumps(CueEvent(done=True).model_dump()) + "\n").encode(
                        "utf-8"
                    )
                return

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _require_job(request: Request, job_id: str) -> Job:
    """Look up ``job_id`` or raise 404.

    Returns:
        The matching :class:`Job`.

    Raises:
        HTTPException:
            404 if no such job exists.
    """
    state: AppState = request.app.state.app_state
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job


def _translate_job(
    job: Job, language: str, llm_client: openai.OpenAI, llm_model: str
) -> None:
    r"""Translate ``job``'s chunks and push :class:`CueEvent`\ s to the job."""

    def _on_batch_done(indices: list[int], translated) -> None:  # noqa: ANN001
        events = [
            CueEvent(
                index=idx,
                start=chunk.start_time,
                end=chunk.end_time,
                text=chunk.text or "",
            )
            for idx, chunk in zip(indices, translated, strict=True)
        ]
        job.append_cues(events)

    try:
        correct_and_translate(
            job.chunks,
            target_language=language,
            client=llm_client,
            model=llm_model,
            batch_size=1,
            on_batch_done=_on_batch_done,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Translation failed for job %s", job.job_id)
        job.mark_done(error=str(exc))
    else:
        job.mark_done()


app = FastAPI(title="drtv_in_english", lifespan=lifespan)
app.include_router(router)
