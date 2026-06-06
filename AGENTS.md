# AGENTS.md

Orientation for AI agents working in this repo.

## What this project is

**DRTV in English** — a Manifest V3 browser extension (Chrome + Firefox)
that injects translated English subtitles into DR's own DRTV player.

The extension is the entire product:

- Content script hooks DR's subtitle button on `dr.dk/drtv/*`.
- Background service worker sniffs the Danish `.vtt`, batches cues,
  calls the user's chosen LLM provider, and streams English cues back.
- English cues are appended to a native `TextTrack` on DR's `<video>`.
- Per-user API key in `chrome.storage.local`. Per-episode translation
  cache in IndexedDB.

No backend, no Docker, no media handling. DR's player handles playback.

## Repo layout

```
manifest.chrome.json
manifest.firefox.json
build.mjs                 # esbuild → dist/chrome and dist/firefox
package.json
tsconfig.json
src/
  background/             # service worker: vtt-sniffer + vtt-parser + translator
  content/                # menu + track injector
  options/                # provider/key form
  shared/                 # types, episode-id helpers, storage wrapper
spike/                    # Phase 0 single-file spike (kept for reference)
icons/
docs/
  extension-plan.md       # full architecture and phased plan
```

## Key files to read first

1. [`docs/extension-plan.md`](docs/extension-plan.md) — architecture,
   decisions, phased plan, lessons learned. **Read before touching
   subtitle/playback code.**
2. [`README.md`](README.md) — what's shipped, what's deferred, how to
   load in each browser.
3. `manifest.chrome.json` / `manifest.firefox.json` — permissions and
   entry points.
4. `src/background/` — service worker, VTT sniffer, parser, translator.
5. `src/content/` — DR-player DOM hooks and the three-way
   Off/Dansk/English menu.

## Common commands

| Task | Command |
| --- | --- |
| Install deps | `npm install` |
| Build (chrome + firefox) | `npm run build` |
| Watch / rebuild on save | `npm run watch` |
| Typecheck | `npm run typecheck` |
| Package for stores | `npm run package` |

## Updating the version

Before releasing, bump the version in **two places**:

1. **`manifest.chrome.json`** — set `"version"` (e.g. `"1.0.0"`)
2. **`manifest.firefox.json`** — set `"version"` (e.g. `"1.0.0"`)

Then run `npm run package` to rebuild with the new version. The build
script reads the version from the manifests and names the output zips
accordingly (e.g. `drtv-in-english-chrome-1.0.0.zip`).

After packaging, update **`CHANGELOG.md`** with the new version number
and release date. Commit all changes together before submitting to stores.

## Things to look out for

- **Lessons from prior iterations.** `docs/extension-plan.md` has a
  section near the top (Lessons from prior iterations) — single attach
  path, don't re-force user state, autoplay must stay inside the user
  gesture, CRLF in VTT will collapse to one cue. Re-deriving these
  costs hours.
- **DRM playback is fine.** Phase 0 confirmed native `TextTrack` works
  on Widevine episodes; no overlay renderer is shipped.
- **MV3 service worker lifetime.** Chrome terminates idle workers
  (~30s). Translation work is chunked into short batches; a port from
  the content script keeps the worker alive while a job runs.
- **API key never leaves the user's machine** except to their
  configured LLM endpoint. Don't add telemetry or remote endpoints
  without an explicit decision in the plan doc.

## What's no longer in this repo

The pre-extension stack (FastAPI backend, Vue 3 + hls.js frontend,
yt-dlp resolver, HLS proxy, Docker Compose, nginx, pytest suite, Python
package, makefile) was removed when we pivoted to the extension. Git
history before that commit still has it.
