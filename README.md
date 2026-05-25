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
URLs are streamed as-is.

## Local Development

```bash
make install
npm run dev                                  # frontend on :5173
uv run fastapi dev src/drtv_in_english/api.py       # backend on :8000
```

The Vite dev server proxies `/api/*` to the backend on port 8000.

## Workflow

```text
 ┌───────────┐  POST /api/prepare       ┌──────────────┐
 │ Frontend  │ ──────────────────────►  │   FastAPI    │
 │ (Vue 3 +  │ ◄──────── job_id ─────── │   backend    │
 │  hls.js)  │                          │              │
 │           │  GET /api/stream/.../    │              │
 │           │  master.m3u8 + segments  │              │
 │           │ ◄──── HLS proxy ──────►  │              │
 │           │                          │              │
 │           │  GET /api/translate/...  │              │
 │           │ ◄──── NDJSON cues ────── │              │
 └───────────┘                          └──────────────┘
```

Nothing is written to disk: the backend resolves DR's HLS playlist and
Danish subtitle URLs with yt-dlp, proxies HLS segments through itself
(re-attaching DR's CDN headers), and streams translated subtitle cues
to the browser as the LLM finishes each batch. The browser shows the
video immediately with Danish subs, then swaps to English as cues
arrive.

## Stack

| Layer | Tech |
| --- | --- |
| Frontend | Vue 3, Vite, TypeScript, hls.js |
| Backend | FastAPI, Pydantic, httpx |
| Translation | OpenAI-compatible LLM (any) |
| Media | `yt-dlp` (metadata only) |
| Infra | Docker Compose, nginx |

## Contribute

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) and the
[code of conduct](./CODE_OF_CONDUCT.md).
