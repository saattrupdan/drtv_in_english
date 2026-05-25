# Browser extension plan — DRTV in English

Replace the current web frontend + backend with a single browser
extension (Chrome + Firefox) that injects translated English subtitles
into DRTV's own video player. Fully serverless: users bring their own
LLM API key.

## Goal

User installs the extension, opens `dr.dk/drtv/episode/…`, presses play
on DRTV's own player, and sees English subtitles appear over the video
within seconds. No separate site, no backend service, no infrastructure
to operate.

## Non-goals

- No proxying of HLS, DRM, or playback. DR's player handles all media.
- No multi-user translation cache. Each user re-translates each episode.
  (Possible later: per-user IndexedDB cache keyed on episode id.)
- No support for non-Chrome/Firefox engines in v1 (Safari MV3 extensions
  use a different distribution path and signing flow — defer).

## Architecture at a glance

```
┌──────────────────────── Extension ────────────────────────┐
│                                                            │
│  content.js  ───────►  background.js  ───────►  LLM API   │
│  (in DRTV    notifies   (service worker)        (OpenAI/   │
│   page)      of episode, parses VTT, calls       Anthropic │
│              streams      LLM in batches,        /etc.)    │
│              cues back    yields cues                      │
│                                                            │
│  options.html  →  chrome.storage.local  (API key, model)  │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      DR's <video> element
                  (cues injected as TextTrack
                   OR as a positioned overlay
                   if DR's custom renderer
                   masks the native track)
```

Single codebase, packaged for two stores. Manifest V3 in both. Firefox
supports MV3 service workers since 121; we can target that.

## Key technical decisions

### 1. Where the translation happens

Background service worker, not the content script. Reasons:

- Service worker stays alive across page navigations within `dr.dk`, so
  partial translation state can persist while the user browses within
  DRTV (within the limits of MV3 service worker lifetime — see risks).
- Cross-origin LLM API calls work from the service worker without
  CORS shenanigans, given the right `host_permissions`.
- Keeps the content script lean and easy to recover if a page reloads.

### 2. How we get the Danish VTT

Three options, in order of preference:

1. **Sniff via `webRequest`** — listen for `*.vtt` responses on DR's CDN,
   capture the URL, then refetch it from the background script. Simple,
   works regardless of DR's player internals.
2. **Read from `performance.getEntries()`** — fallback if `webRequest`
   misses early loads.
3. **Re-resolve via DR's own page data** — the page embeds episode
   metadata in `__NEXT_DATA__` or via XHR; we can grep the subtitle URL
   out of there as a third fallback.

The existing `resolver.py` logic (yt-dlp) cannot run in a browser. The
extension can't use yt-dlp at all. But yt-dlp's job here was mostly URL
extraction — the same data is available client-side via the methods
above.

### 3. How we get the LLM to translate

Reuse the prompt and JSON-output format from `llm.py` exactly. The
extension calls the user's chosen provider directly:

- Default: OpenAI (`https://api.openai.com/v1/chat/completions`),
  configurable to any OpenAI-compatible endpoint (so users can point at
  Anthropic via a proxy, vLLM, Ollama, etc.).
- API key lives in `chrome.storage.local` — never sent anywhere except
  the user's configured endpoint.
- Default `batch_size=1` to match current latency tuning; default
  `max_parallel=20`.

### 4. How we display subtitles

Two-layer approach with runtime fallback:

1. **Primary: native `TextTrack`.** Call
   `videoEl.addTextTrack("subtitles", "English", "en")`, set
   `mode = "showing"`, and `addCue(new VTTCue(...))` as cues stream in.
   This works automatically in fullscreen and uses the browser's own
   cue renderer.
2. **Fallback: positioned overlay div.** If DR's player renders its own
   subtitle DOM and masks the native track, fall back to a fixed-
   position div over the `<video>` element, listening to `timeupdate`
   and looking up the active cue. We already wrote and tested this code
   in an earlier iteration — keep it ready as a runtime fallback.

Detection logic: after attaching the TextTrack, watch for whether
DR's player's `aria-live` subtitle region updates when our cues
activate. If not, switch to overlay mode.

### 5. Manifest V3 for both browsers

- `manifest.json` with a single `background.service_worker` entry.
- `host_permissions`: `https://*.dr.dk/*`, plus the LLM endpoint(s) the
  user configures (use `optional_host_permissions` and request at
  runtime when the key is saved).
- `content_scripts`: match `https://www.dr.dk/drtv/episode/*` and
  inject `content.js` at `document_idle`.
- `web_accessible_resources`: the overlay CSS if we go that route.

Firefox quirks (small):

- Firefox MV3 doesn't fully support
  `chrome.declarativeNetRequest`. We won't need it.
- Firefox uses `browser.*` namespace. Use a small polyfill or wrap with
  `const ext = (globalThis.browser || globalThis.chrome);`.
- Firefox MV3 service workers are real workers since 121; OK to assume.
- Firefox's add-on signing requires submission to AMO; document this in
  README.

## Repo layout (proposed)

```
extension/
  manifest.chrome.json
  manifest.firefox.json
  src/
    background/
      index.ts            # service worker entry
      vtt-sniffer.ts      # webRequest hook
      translator.ts       # LLM call, batching, prompt (port of llm.py)
      vtt-parser.ts       # port of vtt.py
      messaging.ts        # background ↔ content protocol
    content/
      index.ts            # page entry, finds <video>, posts to bg
      track-injector.ts   # adds TextTrack and cues
      overlay-renderer.ts # fallback overlay
    options/
      index.html
      options.ts          # API key + endpoint + model config
    shared/
      types.ts
      storage.ts          # chrome.storage.local wrapper
  build.ts                # esbuild or vite config — produces
                          # extension/dist/chrome and dist/firefox
  package.json
  tsconfig.json
docs/
  extension-plan.md       # this file
```

The existing Python backend stays in place on `main`. The extension
lives in its own subtree so it can be deleted as a unit if abandoned.
The shared logic (prompt text, VTT regex, batching strategy) is
re-implemented in TypeScript rather than shared — porting is a few
hundred lines and avoids a Python ↔ JS build pipeline.

## Phased implementation

### Phase 0 — Spike

Goal: prove the riskiest assumption — that we can inject visible
English text into DR's actual player.

- Hand-write a `manifest.json` and `content.js` that, when loaded on a
  DRTV episode page, finds the `<video>` element and adds one hard-
  coded English `VTTCue` covering 0–60s.
- Verify in both Chrome and Firefox that the cue renders, including in
  fullscreen.
- If it doesn't render via TextTrack, try the overlay approach in the
  same spike.

**Exit criterion:** an English line of text appears over the DRTV
player while the video plays.

### Phase 1 — End-to-end with stubbed translation

- Wire up the real extension skeleton from the layout above.
- `vtt-sniffer` captures DR's Danish VTT URL.
- `vtt-parser` parses it into chunks.
- `translator` *stubs* the LLM: returns "EN: " + Danish text.
- Content script injects all cues.

**Exit criterion:** open a DRTV episode → see "EN: <Danish>" subtitles
appear across the whole episode within a few seconds.

### Phase 2 — Real LLM + options page

- Port the prompt and batching logic from `src/drtv_in_english/llm.py`.
- Build the options page: input for API key, endpoint, model name.
  Persist in `chrome.storage.local`.
- Replace the stub with the real LLM call.
- Add per-batch error handling that falls back to original text (same
  as backend).

**Exit criterion:** real English translation flows through, settings
survive browser restart.

### Phase 3 — Streaming + UX polish

- Stream cues from background → content as each LLM batch finishes
  (don't wait for the whole episode).
- Add a small "Translating… N/M" status pill in the corner of the
  player, hidden once done.
- Overlay fallback path implemented and triggered if TextTrack doesn't
  render.
- Handle SPA navigation: when user clicks to a new episode within DRTV
  without a full page reload, tear down the old job and start a new
  one. Use a `MutationObserver` on the page's URL or
  `chrome.webNavigation.onHistoryStateUpdated`.

**Exit criterion:** good UX on a fresh episode, on episode switches
within DRTV, and on slow networks.

### Phase 4 — Packaging + distribution

- Build both `extension/dist/chrome` (CRX-compatible) and
  `extension/dist/firefox` (XPI-compatible).
- Add a tiny build script that wraps `esbuild` or `vite` and copies the
  right manifest into each output directory.
- Document the install / sideload flow in `extension/README.md`.

**Exit criterion:** unpacked installs from `dist/chrome` and
`dist/firefox` both work; CI builds both zips.

## Risks and unknowns

1. **DR's player may render its own subtitle DOM** and ignore native
   text tracks. Mitigation: overlay fallback already specced. We'll
   know after Phase 0.
2. **DRM (Widevine) content** — most DRTV content is DRM-protected.
   TextTrack injection works regardless (subs are a sibling of video,
   not inside the encrypted stream), but verify in Phase 0 against a
   DRM-protected episode.
3. **MV3 service worker lifetime.** Chrome aggressively terminates
   idle service workers (~30s). Long-running translation jobs may be
   killed mid-flight. Mitigation: chunk the work so each batch
   finishes within seconds; keep a heartbeat to the active tab via
   port messaging to extend the worker's life while a translation is
   in progress.
4. **LLM API CORS.** OpenAI allows browser-origin requests with an
   `Authorization` header. Verify other providers do the same; some
   require a CORS-proxying setup.
5. **API key exposure surface.** The key sits in
   `chrome.storage.local` — accessible to any extension code, not to
   web pages. Acceptable for a personal-use tool; document clearly in
   the options page.
6. **DR may change their VTT URL pattern or move to embedded WebVTT
   inside the HLS manifest.** The `webRequest` sniffer is the most
   robust to URL changes; the `__NEXT_DATA__` fallback is the most
   fragile. We accept that the extension may need maintenance when DR
   changes their stack — that's true of yt-dlp too.

## Open questions for the user

- Do you want a "use a public backend" mode in addition to BYO key, or
  is BYO key the only mode? (BYO key only is the cleanest serverless
  story; the hybrid adds back the ops you're trying to avoid.)
- Default LLM provider — stick with OpenAI compatible? Or also a
  first-class Anthropic provider option?
- Single-action UX (auto-translate every episode) or click-to-activate
  (extension button starts translation per-episode)? Single-action is
  smoother; click-to-activate avoids LLM spend on episodes the user
  ends up skipping.

## What we ditch from the current code

- All of `src/drtv_in_english/api.py`, `hls_proxy.py`, `jobs.py`, and
  the FastAPI lifespan — replaced by the extension's background
  worker.
- `src/drtv_in_english/resolver.py` (yt-dlp) — replaced by client-side
  URL sniffing.
- `src/frontend/` entirely — DRTV's own UI is the frontend.
- Docker, docker-compose, nginx proxy config.
- The Python project would remain as a translation library (could be
  trimmed to `vtt.py`, `llm.py`, `data_models.py`) for anyone who wants
  to run a server, but is no longer part of the user-facing product.

## What we keep

- The translation prompt and batching strategy — proven, just port to
  TypeScript.
- The CRLF-normalising VTT parser — three lines of regex, port verbatim.
- The general UX shape: stream cues as they're ready, don't block on
  the whole episode.
