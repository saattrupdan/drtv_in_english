<!-- markdownlint-disable MD041 -->
# Danglish

Watch DR TV videos with English subtitles.

______________________________________________________________________
[![License](https://img.shields.io/github/license/saattrupdan/danglish)](https://github.com/saattrupdan/danglish/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/saattrupdan/danglish)](https://github.com/saattrupdan/danglish/commits/main)

Developer:

- Dan Saattrup Smart (<dan.smart@alexandra.dk>)

## Quick Start

The fastest way to run the app — frontend, backend, and nginx proxy —
is via Docker Compose:

```bash
docker compose up --build --remove-orphans
```

Then open <http://localhost>, paste a DRTV URL, and click
**Watch with English subs**.

Series URLs (`/drtv/serie/...`) automatically resolve to the first
episode. Single-episode (`/drtv/se/...`) and film (`/drtv/program/...`)
URLs are downloaded as-is.

## Local Development

```bash
make install
npm run dev                                  # frontend on :5173
uv run fastapi dev src/danglish/api.py       # backend on :8000
```

The Vite dev server proxies `/api/*` to the backend on port 8000.

## CLI

For a one-shot run without the web app:

```bash
uv run python src/scripts/run_pipeline.py <DRTV URL> --language en
```

The downloaded `.mp4` and translated `.en.vtt` end up in `./data/`.

## Workflow

```text
 ┌───────────┐   POST /api/process    ┌──────────────┐
 │ Frontend  │ ─────────────────────► │   FastAPI    │
 │ (Vue 3)   │ ◄───────── NDJSON ──── │   backend    │
 └───────────┘   progress events      └──────────────┘
        ▲                                     │
        │  /api/media/<file>                  │
        │  (video + .vtt)                     ▼
        └─────────────────────────────  ./data/
```

Backend pipeline:

```text
 download (yt-dlp)  ──►  parse Danish VTT  ──►  translate with LLM  ──►  write .en.vtt
       0–50%                                          50–100%                100%
```

## Stack

| Layer | Tech |
| --- | --- |
| Frontend | Vue 3, Vite, TypeScript |
| Backend | FastAPI, Pydantic |
| Translation | OpenAI-compatible LLM (any) |
| Media | `yt-dlp`, `ffmpeg` |
| Infra | Docker Compose, nginx |

## Contribute

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) and the
[code of conduct](./CODE_OF_CONDUCT.md).
