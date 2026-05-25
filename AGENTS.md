# AGENTS.md

Orientation for AI agents working in this repo. Python style and tooling
conventions live in the `python` skill — don't duplicate them here.

## What this project is

**Danglish** — watch DRTV with English subtitles. The user pastes a DR TV URL
(series, single episode, or film); the backend resolves it to DR's HLS
playlist + Danish subtitle URL via yt-dlp metadata extraction (no media
downloaded), proxies the HLS stream to the browser with hls.js, and
translates Danish cues with an LLM in the background. The browser shows
the video immediately with Danish subs and swaps to English cue-by-cue
as they arrive over an NDJSON stream.

Nothing is written to disk. DR provides Danish subtitles for everything
on DRTV, so we never run ASR — we just translate the existing subs.

## Stack

- **Python 3.12** package managed with **uv** (`pyproject.toml`, `uv.lock`).
- **OpenAI SDK** for LLM-based translation (`openai>=1.50`). Any
  OpenAI-compatible endpoint works (set `LLM_BASE_URL`, `LLM_API_KEY`,
  `LLM_MODEL`).
- **yt-dlp** for *metadata extraction only* (`extract_info(download=False)`).
  We use it to resolve the HLS master URL, its required request headers,
  and the Danish subtitle URL. Subtitles are picked in priority order
  (`da`, `da_combined`, `da-DK`, …).
- **httpx** in the backend acts as an HLS proxy: re-attaches DR's CDN
  headers to playlist and segment requests and streams bytes back.
- **FastAPI** backend (`src/drtv_in_english/api.py`, served as
  `uvicorn drtv_in_english.api:app`). Endpoints: `GET /health`,
  `POST /prepare`, `GET /stream/{job}/master.m3u8`,
  `GET /stream/{job}/p/{token}`, `GET /subs/{job}/da.vtt`,
  `GET /translate/{job}` (NDJSON of `CueEvent`s).
- **Vue 3 + Vite + TypeScript** frontend in `src/frontend/` with
  **hls.js** for playback. `LandingPageView.vue` is the whole UX: URL
  input → `POST /api/prepare` → `<video>` driven by hls.js with a Danish
  `<track>` while English cues stream in over NDJSON and are appended via
  `TextTrack.addCue()`. The English track is switched on after the first
  cue lands.
- **Docker Compose** with an nginx proxy fronting `frontend` (5173) and
  `backend` (8000). nginx disables `proxy_buffering` on `/api/` so NDJSON
  flushes line-by-line. See `docker-compose.yaml` and
  `docker-compose.nginx.conf`. Vite's dev server mirrors the proxy via
  `server.proxy["/api"]` in `vite.config.ts`.

## Repo layout

```
src/
  drtv_in_english/         # Python package
    __init__.py            # Public surface
    api.py                 # FastAPI app: prepare, HLS proxy, subs, NDJSON translate
    resolver.py            # yt-dlp extract_info (no download): HLS + subtitle URLs
    hls_proxy.py           # m3u8 URI rewriting + opaque token registry
    jobs.py                # In-process Job + JobRegistry (cues + Condition)
    data_models.py         # Pydantic: Chunk, PrepareResponse, CueEvent
    logging_config.py      # configure_logging() + shared package logger
    llm.py                 # build_client() + correct_and_translate()
    vtt.py                 # parse_external_vtt + parse_vtt_text + write_vtt_file
  scripts/
    fix_dot_env_file.py    # Used by `make install` to provision .env
  frontend/
    App.vue, main.ts, routes/, views/LandingPageView.vue

tests/                     # pytest, mirrors src/drtv_in_english/ module names
```

Modules use **relative imports** (`from .foo import bar`); scripts and
tests use **absolute imports** (`from drtv_in_english import bar`).

## End-to-end flow

1. `POST /prepare` — `resolver.resolve()` calls `yt_dlp.extract_info`
   with `download=False`, picks the best HLS-native format and the
   highest-priority Danish `.vtt`, and returns the URLs + the per-request
   `http_headers` DR's CDN expects. The backend fetches the small VTT
   synchronously, parses it via `vtt.parse_vtt_text`, registers a
   `Job` (HLS master URL, header set, cue list, condvar), and spawns a
   daemon thread to translate.
2. The browser plays `GET /stream/{job}/master.m3u8`, which fetches
   DR's master playlist, rewrites every URI inside it (variants and
   `EXT-X-MEDIA URI=` attributes) to opaque `proxy` tokens, and
   registers each upstream URL in the job's `HlsRegistry`. Variant
   playlists are rewritten the same way when the browser follows them.
3. Segment requests hit `GET /stream/{job}/p/{token}`. The proxy looks
   up the upstream URL, reattaches the CDN headers + any `Range` from
   the client, and streams bytes back.
4. The frontend opens `GET /translate/{job}` (NDJSON). Each
   `CueEvent` becomes a `VTTCue` appended via `TextTrack.addCue()`.
   After the first cue lands, the English track flips to `showing` and
   the Danish one to `hidden`. A `done: true` sentinel ends the stream.

`resolver._resolve_episode_url()` does a probe call with
`extract_flat=in_playlist` first. If yt-dlp reports a playlist (i.e. a
series page), we pick the first entry's URL. Single-episode and film
URLs are returned unchanged.

`llm.correct_and_translate()` batches chunks (default `batch_size=5`),
gives each batch a sliding context window of surrounding cues
(`context_window=6`), and runs up to `max_parallel=20` requests
concurrently with a ThreadPoolExecutor. The `on_batch_done` callback is
how cues stream out to the job as each batch finishes. Bad batches fall
back silently to the original Danish text.

## Key files to read first

1. `src/drtv_in_english/__init__.py` — public API surface.
2. `src/drtv_in_english/api.py` — canonical end-to-end path; the `lifespan`
   builds the LLM client + an `httpx.AsyncClient`/`httpx.Client`.
3. `src/drtv_in_english/data_models.py` — every shared Pydantic type.
4. `src/drtv_in_english/resolver.py` — series detection + HLS/subtitle picking.
5. `src/drtv_in_english/hls_proxy.py` — playlist URI rewriting + token registry.
6. `src/drtv_in_english/jobs.py` — `Job` with condvar for cue subscribers.
7. `src/drtv_in_english/llm.py` — translation logic; `on_batch_done` is the
   streaming hook.
8. `src/frontend/views/LandingPageView.vue` — the only frontend view;
   see `attachPlayer` (hls.js + tracks) and `consumeTranslations`
   (NDJSON → `addCue`).
9. `makefile` — canonical commands.

## Things to look out for

- **Streaming contract.** `/translate/{job}` returns NDJSON (one
  `CueEvent` per line, `application/x-ndjson`, terminated by a
  `{"done": true}` line). nginx and Vite both proxy `/api/*` with
  `proxy_buffering off` / HTTP/1.1 so events flush per line. Don't
  reintroduce buffering middleware or batch the generator's yields.
- **HLS proxy is SSRF-safe by construction.** Upstream URLs are never
  embedded in proxy URLs — the browser only sees opaque tokens. The
  proxy will refuse any token not previously registered while parsing
  a playlist. Don't add an "open proxy" endpoint that accepts a URL.
- **Series detection.** `resolver._resolve_episode_url()` is the only
  thing that decides "this is a series" vs "this is an episode". When DR
  changes their URL structure or yt-dlp's extractor changes, this is
  where the breakage lands.
- **No Danish subs = hard error.** `resolve()` raises `ValueError` (→
  HTTP 422) if no Danish subtitle track is listed. DR almost always
  provides subs, but live broadcasts and very recent uploads sometimes
  don't.
- **Job lifetime.** Jobs live in memory for the lifetime of the
  process — no TTL, no GC. One backend process per watcher works fine.
  Don't assume cross-process job sharing.
- **`pytest` treats warnings as errors**
  (`filterwarnings = ["error", ...]` in `pyproject.toml`). New code
  that emits warnings will break CI.

## Common commands

| Task | Command |
| --- | --- |
| Install everything | `make install` |
| Lint + format + type-check | `make check` |
| Run tests | `make test` |
| Build + run via Docker | `make docker` |
| Frontend dev server | `npm run dev` |
| Frontend type-check + lint | `npm run check` |
