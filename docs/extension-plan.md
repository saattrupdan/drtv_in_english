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

## Lessons from prior iterations

These came out of building and tearing down several variants of the
current web frontend. Re-deriving them costs hours; ignoring them
costs real bugs. Read before writing any subtitle / playback code.

- **Exactly one attach path.** When the current frontend had both a
  Vue `watch` and a direct `attachPlayer()` call, the player got
  attached twice, leaving an orphan `<track>` element and a destroyed-
  then-recreated Hls instance. Symptom: "I selected English subs and
  nothing shows." Whatever drives subtitle initialization in the
  extension (the subtitle-button click handler, in our case), make
  sure it runs exactly once per episode load.
- **Don't re-force user state.** We tried `video.addEventListener("volumechange", ...)`
  that re-set `muted = false`, and `textTracks.addEventListener("change", ...)`
  that re-set `mode = "showing"`. In Chrome these "worked"; in Firefox
  the mute button became unclickable (presumably the immediate re-flip
  confused the controls). **Set state once on activation, then trust
  the user's clicks.** If the user mutes after picking English, leave
  them muted.
- **Autoplay after async work won't get audio.** Calling
  `video.play()` from a `.then()` after a fetch + an LLM call is
  outside the user-gesture window. The browser will either reject the
  call or force `muted = true`. The current `Play with English subs`
  button exists for exactly this reason — the play happens inside the
  click handler, audio works. The extension's button-injection plan
  inherits this constraint: the *click* that picks "English" must be
  what eventually triggers playback, even if translation is still
  catching up.
- **CRLF in WebVTT will silently collapse the whole file into one
  cue.** `src/drtv_in_english/vtt.py` normalizes `\r\n` → `\n` before
  the cue regex matches. The regex's lookahead doesn't accept `\r` as
  a line boundary; without normalization, the greedy text capture
  swallows everything to EOF. Symptom: `cue_count == 1` and one giant
  block of English text. Port the normalization line, not just the
  regex.

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
extension calls the user's chosen provider directly. The options page
lets the user pick a provider, which selects both the API shape and a
default endpoint:

| Provider          | API shape           | Default endpoint                                              | Default model       |
| ----------------- | ------------------- | ------------------------------------------------------------- | ------------------- |
| Anthropic         | `/messages`         | `https://api.anthropic.com/v1/messages`                       | `claude-haiku-4-5`  |
| OpenAI            | `/responses`        | `https://api.openai.com/v1/responses`                         | `gpt-5-mini`        |
| Gemini            | `/chat/completions` | `https://generativelanguage.googleapis.com/v1beta/openai`     | `gemini-3.5-flash`  |
| OpenAI-compatible | `/chat/completions` | user-supplied (vLLM, Ollama, OpenRouter, etc.)                | user-supplied       |

One translator module with a thin per-shape adapter — the prompt, the
JSON output schema, and the batching logic are shared; only the
request body, the auth header style (`x-api-key` + `anthropic-version`
for Anthropic, `x-goog-api-key` on the URL or `Authorization: Bearer`
for Gemini's compat endpoint, `Authorization: Bearer` for the rest),
and the response-extraction code differ.

Gemini is a separate preset because it uses `/chat/completions` but
has its own endpoint URL and (for some users) a different auth flow.
The translator code path is the same as any other OpenAI-compatible
backend.

API key lives in `chrome.storage.local` — never sent anywhere except
the user's configured endpoint. Default `batch_size=1` and
`max_parallel=20`, matching the current backend.

### 4. How the user triggers translation

DRTV's existing subtitle button is binary (Danish on / off). We extend
it in-place into a three-way selector — **Off / Dansk / English** —
rather than adding a separate "English subs" button. Picking "English"
is what kicks off translation for the current episode.

Implementation outline:

- Wait for DR's player to mount its controls (MutationObserver on the
  player container).
- Find the existing subtitle/caption button. DR's player is custom
  (Shaka-based), so its DOM is opaque — we identify the button by its
  ARIA label or class prefix and re-test if their build changes.
- Replace its click handler / popup so it opens a small three-item
  menu we render. The menu drives both DR's native track state (for
  Dansk) and our translation pipeline (for English).
- Selection persists for the current episode only — the next episode
  resets to Off unless the user re-picks (avoids surprise LLM spend).
- **Off means no subtitles at all** — we disable both DR's Danish
  track and our injected English track. This is the cleanest semantic
  but does mean the extension writes to DR's player state in both
  directions; on any DR build where toggling their native track from
  outside doesn't stick, we degrade to a warning in the options page
  and let the user toggle Danish via DR's own UI.
- If we can't reliably hook into DR's button (e.g., they wrap it in a
  shadow root or rebuild it on every play/pause), fall back to
  rendering our own pill-shaped "EN" toggle as a sibling, positioned
  next to the existing subtitle button. Still in-player, still one
  click, but a separate widget.

### 5. How we display subtitles

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

### 6. Caching translations locally

The extension keeps an IndexedDB store of completed translations keyed
by DRTV episode id (extracted from the URL — e.g.
`badehoteller_-svinkloev_404057`). On opening an episode the user has
seen before, the cached cues are loaded instantly and no LLM calls
are made.

Cache entry shape:

```ts
{
  episodeId: string,
  sourceVttHash: string,   // sha-256 of the Danish VTT, so cache
                           //   invalidates if DR re-issues subs
  provider: string,        // which LLM made it
  model: string,
  cuesJson: string,        // gzipped or raw, depending on size
  createdAt: number,
}
```

- IndexedDB rather than `chrome.storage.local` because translations can
  be hundreds of KB and `storage.local` has a 10 MB quota.
- Cache check happens *before* fetching the Danish VTT, on the
  episode-id key. If hit, we still confirm `sourceVttHash` matches the
  current VTT to catch DR retranslating their own subs.
- The options page exposes total cache size and a **"Clear cache"**
  button.
- No automatic expiry; users decide when to clear.

### 7. Manifest V3 for both browsers

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

Goal: prove the two riskiest assumptions — that we can render
English text over DR's actual player, and that we can hook into
their subtitle button.

- Hand-write a `manifest.json` and `content.js` that, when loaded on a
  DRTV episode page, finds the `<video>` element and adds one hard-
  coded English `VTTCue` covering 0–60s.
- Verify in both Chrome and Firefox that the cue renders, including in
  fullscreen, on a DRM-protected episode.
- Locate the existing subtitle button in DR's player DOM and verify we
  can either replace its menu or render a sibling widget next to it.
- If TextTrack doesn't render through DR's subtitle layer, try the
  overlay approach in the same spike.

**Exit criterion:** an English line of text appears over the DRTV
player while the video plays, *and* we have a path to a clickable
in-player English-subs control.

### Phase 1 — End-to-end with stubbed translation

- Wire up the real extension skeleton from the layout above.
- Inject the three-way subtitle selector (Off / Dansk / English) into
  DR's player. Picking "English" triggers translation; Off and Dansk
  drive DR's native state.
- `vtt-sniffer` captures DR's Danish VTT URL.
- `vtt-parser` parses it into chunks.
- `translator` *stubs* the LLM: returns "EN: " + Danish text.
- Content script injects all cues.

**Exit criterion:** open a DRTV episode, click "English" in the
subtitle button, see "EN: <Danish>" subtitles appear within a few
seconds.

### Phase 2 — Real LLM + options page

- Port the prompt and batching logic from `src/drtv_in_english/llm.py`.
- Build the options page: provider dropdown (Anthropic / OpenAI /
  OpenAI-compatible), API key, endpoint (auto-filled from provider but
  overridable), model name. Persist in `chrome.storage.local`.
- Implement three thin adapters (`/messages`, `/responses`,
  `/chat/completions`) behind a shared `translator` interface.
- Replace the stub with the real LLM call.
- Per-batch error handling that falls back to original text (same as
  backend).

**Exit criterion:** real English translation flows through against all
three API shapes; settings survive browser restart.

### Phase 2b — Caching

- Add IndexedDB cache keyed by `(episodeId, sourceVttHash)`.
- On episode load, hit cache before fetching VTT.
- Add cache-size readout and clear-cache button to the options page.

**Exit criterion:** re-watching a translated episode causes zero LLM
calls.

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

## Decisions

- **BYO key only.** No public backend mode. Single user → single key →
  single LLM bill.
- **Four provider presets** (Anthropic, OpenAI, Gemini, OpenAI-compatible)
  fronting three API shapes (`/messages`, `/responses`,
  `/chat/completions`). Each preset ships a default endpoint + default
  model; both are editable in the options page.
- **Default models:** Anthropic → `claude-haiku-4-5`,
  OpenAI → `gpt-5-mini`, Gemini → `gemini-3.5-flash`,
  OpenAI-compatible → user-supplied.
- **Click-to-activate per episode** via the in-player subtitle button.
  The injected menu is Off / Dansk / English; picking English triggers
  translation. Selection does not persist across episodes (avoids
  surprise LLM spend).
- **Off means no subtitles** — both Danish and English are turned off
  when the user picks Off in our menu.
- **Sibling pill fallback.** If DR's existing subtitle button cannot
  be hooked, we render a "EN" pill next to it instead. Still in-player,
  still one click.
- **Local cache** in IndexedDB, keyed by episode id and source-VTT
  hash. Clear-cache button in the options page; no auto-expiry.

## Open questions for the project owner

These can't be answered by reading the code or this plan — they
depend on the owner's intent. Whoever picks this up: get answers
before starting Phase 4 (and ideally before Phase 0).

- **Sideload only, or publish to stores?** If sideload (personal /
  small group), most of Phase 4 is a 10-minute zip. If publishing to
  Chrome Web Store + Firefox AMO, Phase 4 grows considerably:
  privacy policy URL, per-permission justifications, multi-resolution
  icons, screenshots, store listing copy, and Firefox add-on signing
  via AMO submission.
- **A reliable test-episode URL for Phase 0.** Needs to be: freely
  available in the relevant region, currently in DR's catalogue (not
  rotated out), DRM-protected (most DR content is, and we have to
  prove the approach works there), and has Danish subtitles.

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
