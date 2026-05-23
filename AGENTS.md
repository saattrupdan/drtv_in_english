# AGENTS.md

Orientation for AI agents working in this repo. Python style, docstring conventions,
linter setup, and similar generic guidance live in the `python` skill — don't duplicate
them here.

## What this project is

**But With Subs** — watch anything online "but with subs." The user provides a video
URL (e.g. a DR TV / streaming page); the system downloads the media, transcribes it,
optionally translates the transcript, and produces a WebVTT subtitle file. Speech
recognition is currently Danish-first (`CoRal-project/roest-v3-wav2vec2-315m`).

## Stack

- **Python 3.12** package managed with **uv** (`pyproject.toml`, `uv.lock`).
- **PyTorch + Hugging Face transformers** for ASR (`wav2vec2`) and translation
  (`alirezamsh/small100` is replaced by an OpenAI-compatible LLM client).
- **OpenAI SDK** for LLM-based correction and translation (`openai>=1.50`).
- **pyannote.audio 4.x** for VAD and speaker diarization
  (`speaker-diarization-community-1`).
- **punctfix** for restoring punctuation after ASR.
- **yt-dlp** for downloading source media.
- **FastAPI** backend (`src/but_with_subs/api.py`, served as
  `uvicorn but_with_subs.api:app`). Exposes `GET /health`, `POST /process` (streams
  NDJSON `ProgressEvent`s through the pipeline), and a `/media` static mount serving
  files from `./data/`.
- **SQLModel + Postgres** for persisting URL → media/subtitles mappings
  (`database.py`). Connection string comes from `DATABASE_URL`; falls back to a local
  sqlite file when unset so the API also runs outside Docker.
- **Vue 3 + Vite + TypeScript** frontend in `src/frontend/`. `LandingPageView.vue`
  drives the whole UX: URL/file input → real `POST /api/process` call → live progress
  bar from the streamed NDJSON → HTML5 `<video>` with a `<track kind="subtitles">`.
- **Docker Compose** with an nginx proxy fronting `frontend` (5173), `backend` (8000),
  and `postgres:18.3-trixie`. nginx disables `proxy_buffering` on `/api/` and uses 1-h
  read/send timeouts so the NDJSON stream flushes line-by-line. See
  `docker-compose.yaml` + `docker-compose.nginx.conf`. Vite's dev server mirrors the
  proxy via `server.proxy["/api"]` in `vite.config.ts`.

## Repo layout

```
src/
  but_with_subs/          # Python package — the "modules"
    __init__.py           # Public surface: download, configure_logging, generate_subtitles
    api.py                # FastAPI app: /health, POST /process (NDJSON stream), /media static mount
    pipeline.py           # run_pipeline() generator — orchestrates the API's processing stages
    database.py           # SQLModel FileRecord + build_engine/init_db/upsert_file (Postgres/sqlite)
    constants.py          # Model IDs, sample rate, language codes, palette, DATA_DIR
    data_models.py        # Pydantic models (Chunk, File, DownloadProgress, ProgressEvent, VideoWithSubs)
    logging_config.py     # configure_logging() + shared package logger
    device.py             # get_device() — cuda / mps / cpu selection
    downloading.py        # yt-dlp wrapper, yields DownloadProgress
    audio_extraction.py   # ffmpeg-based video→audio
    audio_loading.py      # load wav to mono 16 kHz numpy, with preprocessing
    audio_chunking.py     # Diarization-based chunking (pyannote)
    transcribing.py       # VAD-segment + wav2vec2 ASR → word-level chunks
    text_chunking.py      # Group word-level chunks into readable subtitle lines
    llm.py                # LLM-based correct_and_translate() + build_client()
    subtitling.py         # generate_subtitles() — writes WebVTT, colours overlapping speakers
    vtt.py                # WebVTT timestamp formatting
  scripts/                # CLI entry points (Click); imported via `from but_with_subs import ...`
    download_video.py
    extract_audio.py
    chunk_audio.py
    transcribe_audio.py   # ASR + grouping + LLM correct-and-translate on an existing .wav
    run_pipeline.py       # End-to-end: URL → download → extract → transcribe → translate → .vtt
    fix_dot_env_file.py   # Used by `make install` to provision .env
  frontend/               # Vue 3 SPA
    App.vue, main.ts, routes/, views/LandingPageView.vue  # Full UI: URL/file input,
                                                          # streaming progress, <video>+<track>

tests/                    # pytest, mirrors src/but_with_subs/ module names
data/                     # Downloaded media + generated .vtt outputs (gitignored content)
```

Modules use **relative imports** (`from .foo import bar`); scripts and tests use
**absolute imports** (`from but_with_subs import bar`). Don't mix them.

## End-to-end flow

There are two end-to-end entry points that share the same building blocks:

- **`src/but_with_subs/pipeline.py:run_pipeline()`** — the canonical path. A generator
  that yields `ProgressEvent`s (stage, percentage 0-100, message, optional `result`)
  and is consumed by `POST /process` as an NDJSON `StreamingResponse`. Heavy ML models
  are passed in (loaded once at FastAPI startup via `lifespan`), not constructed here.
  Stage boundaries: download 0-50, transcribe 50-95, subtitle 95-100. The final
  `completed` event carries a `VideoWithSubs(video_path, subtitles_path)` whose paths
  are rewritten by `api.py:_to_media_url` to `/api/media/<filename>` so the browser
  can fetch them through the same proxy. A `FileRecord` row is upserted after
  download and again after subtitles are generated.
- **`src/scripts/run_pipeline.py`** — CLI wrapper for local one-shot runs without
  the API. Composes the same modules but writes `.vtt` files directly. Useful for
  debugging individual stages.

The CLI path also composes two earlier per-stage scripts:

**1. `src/scripts/download_video.py`** — `download(url, progress_hook=...)` from
`downloading.py` wraps `yt-dlp`:

  - creates `./data/` if missing
  - downloads `bestvideo*+bestaudio*` into `DATA_DIR` (no playlists)
  - streams `DownloadProgress` updates through the hook (driven by yt-dlp's
    `progress_hooks`)
  - scans `./data/` for the first video file (`.mp4/.webm/.mkv/.avi/.mov`) and audio
    file (`.mp3/.m4a/.wav/.flac/.aac/.ogg`), and returns a `File(url, video_path,
    audio_path)` model

**2. `src/scripts/transcribe_audio.py`** — takes the `.wav` produced by
`extract_audio()` (ffmpeg) and runs:

  1. Load the ASR pipeline (`ASR_MODEL_ID`, `num_beams=5`) on the device from
     `get_device()`.
  2. Load `PunctFixer(language="da")` for Danish punctuation restoration.
  3. `load_audio(path)` → float32 mono 16 kHz numpy array (with preprocessing).
  4. `transcribe_audio(audio, model)` in `transcribing.py` — VAD-segments the audio
     (pyannote VAD) and runs wav2vec2 over each segment to produce **word-level**
     chunks with timestamps.
  5. `group_word_chunks(word_chunks, punctuation_model, max_words=MAX_WORDS)` in
     `text_chunking.py` — joins words into readable subtitle segments and restores
     punctuation.
  6. `correct_and_translate(chunks, target_language, client, model)` in `llm.py` —
     LLM-based ASR correction and translation with sliding context window; returns
     updated Chunk objects.
  7. `generate_subtitles(...)` in `subtitling.py` writes a single `.vtt` file
     (`.<lang>.vtt`), colour-coding overlapping speakers.

**`run_pipeline.py`** glues these together: download → `extract_audio(video_path)` →
transcribe + translate → write `.vtt`s → delete the intermediate `.wav` in a `finally`
block (the source video stays in `./data/`).

## Key files to read first

1. `src/but_with_subs/__init__.py` — the public API surface.
2. `src/but_with_subs/api.py` + `src/but_with_subs/pipeline.py` — the canonical
   end-to-end path; the API's `lifespan` shows how models are loaded once and the
   pipeline shows how they're stitched together.
3. `src/but_with_subs/constants.py` — model IDs and tunables (`ASR_MODEL_ID`,
   `TARGET_SAMPLE_RATE`, `MAX_WORDS`, `DATA_DIR`).
4. `src/but_with_subs/data_models.py` — every shared Pydantic / data type.
5. `src/but_with_subs/database.py` — `FileRecord` schema and `upsert_file` semantics
   (partial updates).
6. `src/frontend/views/LandingPageView.vue` — the only frontend view; see
   `consumeStream` for the NDJSON parsing contract.
7. `src/scripts/run_pipeline.py` — CLI variant of the pipeline.
8. `makefile` — canonical commands.

## Things to look out for

- **Streaming contract.** `/process` returns NDJSON (one `ProgressEvent` per line,
  `application/x-ndjson`). nginx (`docker-compose.nginx.conf`) and Vite
  (`vite.config.ts`) both proxy `/api/*` to the backend with `proxy_buffering off` /
  HTTP/1.1 so events flush per line. Don't reintroduce buffering middleware or batch
  the generator's yields.
- **Path → URL rewriting.** The pipeline emits absolute filesystem paths for
  `VideoWithSubs.video_path` / `subtitles_path`; `api.py:_to_media_url` rewrites them
  to `/api/media/<filename>` against the `/media` static mount. If you add fields with
  paths, rewrite them here too — the browser can't read absolute container paths.
- **Models load once at startup.** `lifespan` builds the ASR pipeline, `PunctFixer`,
  LLM client, and database engine into `app.state.app_state`. Don't re-instantiate
  them per request, and don't import `pipeline.run_pipeline` in a way that forces
  eager model loading.
- **Database.** `database.build_engine()` reads `DATABASE_URL`
  (`postgresql+psycopg://…` in Docker, sqlite fallback elsewhere). `psycopg[binary]`
  is the driver. `upsert_file` does **partial** updates — `None` arguments don't
  overwrite — so it's safe to set `subtitles_path` after the initial insert.
- **ffmpeg in the backend image.** `Dockerfile.backend` apt-installs `ffmpeg`
  because `yt-dlp` and `audio_extraction.py` shell out to it. Don't drop this when
  refactoring the image.
- **pyannote.audio 4.x quirks**: there's a history of fixes around the VAD API
  (`instantiate()` for threshold params, calling `.apply()` instead of `__call__`).
  See commits `90e0d59`, `220cdf7`, `27b43ef`. Check current pyannote.audio docs before
  changing `transcribing.py:vad_segment_audio` or `audio_chunking.py`.
- **Heavy model downloads** on first run (wav2vec2 ~315M, pyannote diarization,
  pyannote VAD). Tests mock `vad_segment_audio` to avoid this — preserve that
  pattern (see commit `b6b9d2c`). Some models (pyannote diarization-community-1)
  require accepting the Hugging Face license + a HF token. The LLM client is
  API-based and requires no local model download.
- **Audio is float32, mono, 16 kHz, contiguous** by the time it hits ASR. The
  preprocessing/normalisation invariants in `audio_loading.py` matter — see fixes in
  commits `6a5378a`, `b078600`.
- **Danish-centric defaults**: ASR model is Danish-only (`CoRal` Roest), and
  `transcribe_audio.py` hard-codes `PunctFixer(language="da")`. Generalising to other
  source languages requires touching both.
- **`numpy>=2.0` is force-overridden** via `[tool.uv] override-dependencies` because
  some transitive deps pin older numpy. Don't undo this without checking.
- **Pydantic models with `np.ndarray`** (`Chunk`) require
  `model_config = {"arbitrary_types_allowed": True}`. Keep that when adding similar
  models.
- **Speaker colours** in `subtitling.py` are intentionally colourblind-safe (Wong
  palette). Don't replace with arbitrary hex codes.
- **`pytest` treats warnings as errors** (`filterwarnings = ["error", ...]` in
  `pyproject.toml`). New code that emits warnings will break CI.
- `data/` contains real downloaded media (Danish TV); content is gitignored but the
  directory exists locally — don't accidentally commit large media.

## Common commands

| Task | Command |
| --- | --- |
| Install everything | `make install` |
| Lint + format + type-check | `make check` |
| Run tests + update coverage badge | `make test` |
| Build + run via Docker | `make docker` |
| Full pipeline (URL → subtitles) | `uv run python src/scripts/run_pipeline.py <URL> --language en` |
| Run a script | `uv run python src/scripts/<name>.py --help` |
| Frontend dev server | `npm run dev` |
| Frontend type-check + lint | `npm run check` |
