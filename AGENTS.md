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
  (`alirezamsh/small100`, an M2M100 variant; ships with a custom tokenizer in
  `tokenization_small100.py`).
- **pyannote.audio 4.x** for VAD and speaker diarization
  (`speaker-diarization-community-1`).
- **punctfix** for restoring punctuation after ASR.
- **yt-dlp** for downloading source media.
- **FastAPI** backend (`src/but_with_subs/api.py`, served as
  `uvicorn but_with_subs.api:app`). Currently just a `/health` stub — real endpoints
  still need to be built.
- **Vue 3 + Vite + TypeScript** frontend in `src/frontend/`. Currently a stub —
  `LandingPageView.vue` is a single `<h1>` placeholder.
- **Docker Compose** with an nginx proxy fronting `frontend` (port 5173) and `backend`
  (port 8000). See `docker-compose.yaml` + `docker-compose.nginx.conf`.

## Repo layout

```
src/
  but_with_subs/          # Python package — the "modules"
    __init__.py           # Public surface: download, configure_logging, generate_subtitles
    api.py                # FastAPI app (stub; add endpoints here)
    constants.py          # Model IDs, sample rate, language codes, palette
    data_models.py        # Centralised Pydantic models (Chunk, File, DownloadProgress)
    logging_config.py     # configure_logging() + shared package logger
    device.py             # get_device() — cuda / mps / cpu selection
    downloading.py        # yt-dlp wrapper, yields DownloadProgress
    audio_extraction.py   # ffmpeg-based video→audio
    audio_loading.py      # load wav to mono 16 kHz numpy, with preprocessing
    audio_chunking.py     # Diarization-based chunking (pyannote)
    transcribing.py       # VAD-segment + wav2vec2 ASR → word-level chunks
    text_chunking.py      # Group word-level chunks into readable subtitle lines
    translation.py        # Batched small100 translation of Chunks
    tokenization_small100.py  # Vendored tokenizer for the small100 model
    subtitling.py         # generate_subtitles() — writes WebVTT, colours overlapping speakers
    vtt.py                # WebVTT timestamp formatting
  scripts/                # CLI entry points (Click); imported via `from but_with_subs import ...`
    download_video.py
    extract_audio.py
    chunk_audio.py
    transcribe_audio.py   # ASR + grouping + translation on an existing .wav
    translate_string.py
    run_pipeline.py       # End-to-end: URL → download → extract → transcribe → translate → .vtt
    fix_dot_env_file.py   # Used by `make install` to provision .env
  frontend/               # Vue 3 SPA (currently stub)
    App.vue, main.ts, routes/, views/LandingPageView.vue

tests/                    # pytest, mirrors src/but_with_subs/ module names
data/                     # Downloaded media + generated .vtt outputs (gitignored content)
```

Modules use **relative imports** (`from .foo import bar`); scripts and tests use
**absolute imports** (`from but_with_subs import bar`). Don't mix them.

## End-to-end flow

The full URL-to-subtitles pipeline lives in `src/scripts/run_pipeline.py`. It composes
two earlier per-stage scripts:

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
  6. `translate_chunks(chunks, target_lang, batch_size=16)` in `translation.py` —
     batched small100 translation; yields per-chunk progress tuples then translated
     `Chunk`s.
  7. `generate_subtitles(...)` in `subtitling.py` writes two `.vtt` files: `.da.vtt`
     (source) and `.<lang>.vtt` (translated), colour-coding overlapping speakers.

**`run_pipeline.py`** glues these together: download → `extract_audio(video_path)` →
transcribe + translate → write `.vtt`s → delete the intermediate `.wav` in a `finally`
block (the source video stays in `./data/`).

## Key files to read first

1. `src/but_with_subs/__init__.py` — the public API surface.
2. `src/but_with_subs/constants.py` — model IDs and tunables (`ASR_MODEL_ID`,
   `TRANSLATION_MODEL`, `TARGET_SAMPLE_RATE`, `MAX_WORDS`).
3. `src/but_with_subs/data_models.py` — every shared Pydantic / data type.
4. `src/scripts/run_pipeline.py` — the canonical end-to-end pipeline (URL →
   `.vtt`); read this before the per-stage scripts.
5. `makefile` — canonical commands.

## Things to look out for

- **Backend is a stub.** `src/but_with_subs/api.py` only exposes `/health` so the
  Docker build succeeds. Real endpoints (download, transcribe, subtitle) still need to
  be wired up before the frontend can talk to anything.
- **pyannote.audio 4.x quirks**: there's a history of fixes around the VAD API
  (`instantiate()` for threshold params, calling `.apply()` instead of `__call__`).
  See commits `90e0d59`, `220cdf7`, `27b43ef`. Check current pyannote.audio docs before
  changing `transcribing.py:vad_segment_audio` or `audio_chunking.py`.
- **Heavy model downloads** on first run (wav2vec2 ~315M, small100, pyannote
  diarization, pyannote VAD). Tests mock `vad_segment_audio` to avoid this — preserve
  that pattern (see commit `b6b9d2c`). Some models (pyannote diarization-community-1)
  require accepting the Hugging Face license + a HF token.
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
