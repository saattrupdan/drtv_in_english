<!-- This disables the "First line in file should be a top level heading" rule -->
<!-- markdownlint-disable MD041 -->
<p align="center">
  <img
    src="./public/but-with-subs-logo.jpg"
    width="240"
    alt="But with subs logo"
  />
</p>

# But With Subs

Watch anything online... but with subs.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-95%25-brightgreen.svg)](https://github.com/saattrupdan/but_with_subs/tree/main/tests)
[![License](https://img.shields.io/github/license/saattrupdan/but_with_subs)](https://github.com/saattrupdan/but_with_subs/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/but_with_subs)](https://github.com/saattrupdan/but_with_subs/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/saattrupdan/but_with_subs/blob/main/CODE_OF_CONDUCT.md)

Developer:

- Dan Saattrup Smart (<dan.smart@alexandra.dk>)

## Quick Start

The fastest way to run the full app — frontend, backend, postgres, and the
nginx reverse proxy — is via Docker Compose:

```bash
docker compose up --build --remove-orphans
```

Once the containers are healthy, open <http://localhost> in your browser,
paste a video URL, and click **Watch with Subs**.

## Local Development

For iterating on the code without Docker, run the frontend and backend
separately.

1. Install everything:

   ```bash
   make install
   ```

2. Start the frontend dev server:

   ```bash
   npm run dev
   ```

3. In a second terminal, start the backend with hot reload:

   ```bash
   uv run fastapi dev src/but_with_subs/api.py
   ```

The Vite dev server proxies `/api/*` to the backend on port 8000, so the
frontend at <http://localhost:5173> talks to the API exactly the way it
does behind nginx in production.

## Workflow

When a user submits a URL, the request fans out across the stack:

```text
 ┌───────────┐   POST /api/process    ┌──────────────┐
 │ Frontend  │ ─────────────────────► │   FastAPI    │
 │ (Vue 3)   │ ◄───────── NDJSON ──── │   backend    │
 └─────┬─────┘   progress events      └──────┬───────┘
       │                                     │
       │  /api/media/<file>                  │ upsert
       │  (video + .vtt)                     ▼
       │                              ┌──────────────┐
       └─────────────────────────────►│  Postgres    │
                                      └──────────────┘
```

The backend pipeline is a single generator that yields
`ProgressEvent` objects as it works through five stages:

```text
 download ──► extract audio ──► VAD + transcribe ──► (translate?) ──► subtitle
   0–50%           │                 50–95%             95%             100%
                   └─► upsert FileRecord(url, video_path, audio_path)
                                                                       │
                                                       upsert subtitles_path
```

The final event carries a `VideoWithSubs` payload with browser-facing
URLs (`/api/media/<filename>`); the frontend wires those into the
`<video>` element and a `<track kind="subtitles">`.

## Stack

| Layer | Tech |
| --- | --- |
| Frontend | Vue 3, Vite, TypeScript |
| Backend | FastAPI, SQLModel, Pydantic |
| Database | Postgres 18.3 |
| ML | `transformers` (Wav2Vec2 ASR), `pyannote.audio` (VAD), `punctfix`, `M2M100` translation |
| Media | `yt-dlp`, `ffmpeg` |
| Infra | Docker Compose, nginx reverse proxy |

Heavy ML models are loaded once at FastAPI startup via the `lifespan`
context and shared across requests.

## Contribute

Contributions are very welcome. See [`CONTRIBUTING.md`](./CONTRIBUTING.md)
for the workflow, coding conventions, and the Contributor Covenant
[code of conduct](./CODE_OF_CONDUCT.md).
