# AGENTS.md

Orientation for AI agents working in this repo. Python style and tooling
conventions live in the `python` skill — don't duplicate them here.

## What this project is

**Danglish** — watch DRTV with English subtitles. The user pastes a DR TV URL
(series, single episode, or film); the system downloads the video + DR's
own Danish subtitle track, translates the subtitles via an LLM, and serves
the video with the translated `<track>` in the browser.

DR provides Danish subtitles for everything on DRTV, so we never run ASR —
we just translate the existing subs.

## Stack

- **Python 3.12** package managed with **uv** (`pyproject.toml`, `uv.lock`).
- **OpenAI SDK** for LLM-based translation (`openai>=1.50`). Any
  OpenAI-compatible endpoint works (set `LLM_BASE_URL`, `LLM_API_KEY`,
  `LLM_MODEL`).
- **yt-dlp** for downloading. Subtitles are requested in priority order
  (`da`, `da_combined`, `da-DK`, …) and the highest-priority track is
  picked. ffmpeg is needed for muxing.
- **FastAPI** backend (`src/danglish/api.py`, served as
  `uvicorn danglish.api:app`). Endpoints: `GET /health`,
  `POST /process` (streams NDJSON `ProgressEvent`s), `/media` static mount
  serving files from `./data/`.
- **Vue 3 + Vite + TypeScript** frontend in `src/frontend/`.
  `LandingPageView.vue` is the whole UX: URL input → real
  `POST /api/process` call → streaming progress bar → HTML5 `<video>` with
  the English `<track>`.
- **Docker Compose** with an nginx proxy fronting `frontend` (5173) and
  `backend` (8000). nginx disables `proxy_buffering` on `/api/` so NDJSON
  flushes line-by-line. See `docker-compose.yaml` and
  `docker-compose.nginx.conf`. Vite's dev server mirrors the proxy via
  `server.proxy["/api"]` in `vite.config.ts`.

## Repo layout

```
src/
  danglish/                # Python package
    __init__.py            # Public surface
    api.py                 # FastAPI app: /health, POST /process (NDJSON), /media
    pipeline.py            # run_pipeline() generator — download → translate → write
    constants.py           # DATA_DIR, TARGET_SAMPLE_RATE
    data_models.py         # Pydantic: Chunk, File, ProgressEvent, VideoWithSubs,
                           # DownloadProgress
    logging_config.py      # configure_logging() + shared package logger
    downloading.py         # yt-dlp wrapper. Series URLs auto-resolve to first episode.
    llm.py                 # build_client() + correct_and_translate()
    vtt.py                 # parse_external_vtt + write_vtt_file + ts helpers
  scripts/
    run_pipeline.py        # CLI: DR URL → translated .vtt
    fix_dot_env_file.py    # Used by `make install` to provision .env
  frontend/
    App.vue, main.ts, routes/, views/LandingPageView.vue

tests/                     # pytest, mirrors src/danglish/ module names
data/                      # Downloaded media + generated .vtt (gitignored)
```

Modules use **relative imports** (`from .foo import bar`); scripts and
tests use **absolute imports** (`from danglish import bar`).

## End-to-end flow

Two entry points share `download → parse_external_vtt → correct_and_translate
→ write_vtt_file`:

- **`src/danglish/pipeline.py:run_pipeline()`** — generator yielding
  `ProgressEvent`s. Consumed by `POST /process` as an NDJSON
  `StreamingResponse`. Stage boundaries: download 0-50, translate 50-100.
  Final `completed` event carries a `VideoWithSubs(video_path,
  subtitles_path)`; `api.py:_to_media_url` rewrites the absolute
  filesystem paths to `/api/media/<filename>` so the browser can fetch
  them through the same proxy.
- **`src/scripts/run_pipeline.py`** — CLI wrapper that writes `.vtt`
  files directly without going through the API.

`downloading.download()` does a probe call with `extract_flat=in_playlist`
first. If yt-dlp reports a playlist (i.e. a series page), we pick the
first entry's URL and download that. Single-episode and film URLs return
unchanged from the probe.

`llm.correct_and_translate()` batches chunks (default `batch_size=5`),
gives each batch a sliding context window of surrounding cues
(`context_window=6`), and runs up to `max_parallel=20` requests
concurrently with a ThreadPoolExecutor. Bad batches fall back silently
to the original text.

## Key files to read first

1. `src/danglish/__init__.py` — public API surface.
2. `src/danglish/api.py` + `src/danglish/pipeline.py` — canonical
   end-to-end path. `lifespan` just builds the LLM client.
3. `src/danglish/data_models.py` — every shared Pydantic type.
4. `src/danglish/downloading.py` — series detection lives here.
5. `src/danglish/llm.py` — translation logic and batching strategy.
6. `src/frontend/views/LandingPageView.vue` — the only frontend view; see
   `consumeStream` for the NDJSON parsing contract.
7. `src/scripts/run_pipeline.py` — CLI variant.
8. `makefile` — canonical commands.

## Things to look out for

- **Streaming contract.** `/process` returns NDJSON (one `ProgressEvent`
  per line, `application/x-ndjson`). nginx and Vite both proxy `/api/*`
  with `proxy_buffering off` / HTTP/1.1 so events flush per line. Don't
  reintroduce buffering middleware or batch the generator's yields.
- **Path → URL rewriting.** The pipeline emits absolute filesystem paths
  for `VideoWithSubs.video_path` / `subtitles_path`; `api.py:_to_media_url`
  rewrites them to `/api/media/<filename>` against the `/media` static
  mount. If you add fields with paths, rewrite them here too.
- **Series detection.** `downloading._resolve_episode_url()` is the only
  thing that decides "this is a series" vs "this is an episode". When DR
  changes their URL structure or yt-dlp's extractor changes, this is
  where the breakage lands.
- **No Danish subs = hard error.** The pipeline yields a
  `stage="error"` event if `file.subtitles_path is None`. DR almost
  always provides subs, but live broadcasts and very recent uploads
  sometimes don't.
- **ffmpeg in the backend image.** `Dockerfile.backend` apt-installs
  ffmpeg because yt-dlp shells out to it during muxing. Don't drop this.
- **`pytest` treats warnings as errors**
  (`filterwarnings = ["error", ...]` in `pyproject.toml`). New code
  that emits warnings will break CI.
- `data/` contains real downloaded media (Danish TV); content is
  gitignored but the directory exists locally — don't commit large media.

## Common commands

| Task | Command |
| --- | --- |
| Install everything | `make install` |
| Lint + format + type-check | `make check` |
| Run tests | `make test` |
| Build + run via Docker | `make docker` |
| Full pipeline (URL → subtitles) | `uv run python src/scripts/run_pipeline.py <DR URL> --language en` |
| Frontend dev server | `npm run dev` |
| Frontend type-check + lint | `npm run check` |
