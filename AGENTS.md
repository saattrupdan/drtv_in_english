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

## Build artifacts: `dist/` vs `package/`

- **`dist/`** is throwaway build output (gitignored). `npm run build`
  writes the unpacked extensions to `dist/chrome` and `dist/firefox`;
  `npm run package` also drops store zips in `dist/`. **Load `dist/` for
  dev** (Load unpacked / Load Temporary Add-on).
- **`package/`** holds the *committed* store artifacts only:
  `package/zips/` (the exact zips submitted to the stores),
  `package/assets/` (screenshots), `package/submission/` (listing text,
  permission justifications, source-submission notes).
- **Do NOT commit `package/chrome/` or `package/firefox/`** — those
  unpacked build trees are gitignored. They used to be committed and
  drifted out of sync with `dist/`, causing "fixed it but it's still
  broken" confusion (a stale tree was being loaded). Build to `dist/`;
  copy zips into `package/zips/` for the release record.

## Updating the version

Before releasing, bump the version in **three places** (keep them in
sync):

1. **`manifest.chrome.json`** — set `"version"` (e.g. `"1.0.1"`)
2. **`manifest.firefox.json`** — set `"version"`
3. **`package.json`** — set `"version"`. `build.mjs` reads the version
   from **`package.json`** (not the manifests) when naming the zips, so
   if you skip this the zips get the wrong/stale name.

Then run `npm run package` to rebuild and zip. Copy the resulting
`dist/drtv-in-english-{chrome,firefox}-<version>.zip` into
`package/zips/` (removing the previous version's zips).

For a **Firefox AMO** update you also need a source zip (Mozilla
requires it for bundled add-ons). Generate one from the repo source:

```
zip -r -q -X dist/drtv-in-english-source-<version>.zip \
  src build.mjs package.json package-lock.json tsconfig.json \
  manifest.chrome.json manifest.firefox.json icons \
  README.md AGENTS.md CHANGELOG.md -x '*.DS_Store'
```

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
