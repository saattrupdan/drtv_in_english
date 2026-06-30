# Permission Justifications

## `host_permissions`

### `https://www.dr.dk/*`
Read `<video>` elements, inject subtitle tracks, and observe DR's subtitle network requests on the DRTV player pages.

### `https://api.anthropic.com/*`, `https://api.openai.com/*`, `https://generativelanguage.googleapis.com/*`, `https://inference.alexandra.dk/*`
The endpoints of the built-in LLM provider presets. Translation requests are sent directly to whichever provider the user configures, with the user's own API key. Pre-listing the built-in providers lets them work without an extra permission prompt; no other data is sent to these hosts.

## `optional_host_permissions`

### `*://*/*`
Requested at runtime, scoped to the specific host, when the user saves a **custom** LLM endpoint (the "OpenAI-compatible" provider). Used to send translation requests to that user-supplied endpoint. The broad pattern only defines what *may* be requested — the extension never requests it wholesale; it requests access to the exact host the user enters. Both schemes are allowed so locally hosted models (e.g. `http://localhost:8080`) work alongside remote `https://` endpoints. No custom service URLs are hardcoded.

## `permissions`

### `webRequest`
Capture the Danish `.vtt` subtitle URL from DR's video player network requests.

### `storage`
Store the user's API key and per-episode translation cache metadata in `chrome.storage.local`.

### `webNavigation`
Detect single-page-application (SPA) navigation between DRTV episodes to trigger per-episode translation caching.
