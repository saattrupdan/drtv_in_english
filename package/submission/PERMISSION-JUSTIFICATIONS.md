# Permission Justifications

## `host_permissions`

### `https://www.dr.dk/*`
Read `<video>` elements, inject subtitle tracks, and observe DR's subtitle network requests on the DRTV player pages.

## `optional_host_permissions`

### `https://*/`
Requested at runtime when the user saves their LLM API key. Used to send translation requests to the user's chosen LLM endpoint. The user selects and configures this endpoint; the extension does not hardcode any third-party service URLs.

## `permissions`

### `webRequest`
Capture the Danish `.vtt` subtitle URL from DR's video player network requests.

### `storage`
Store the user's API key and per-episode translation cache metadata in `chrome.storage.local`.

### `webNavigation`
Detect single-page-application (SPA) navigation between DRTV episodes to trigger per-episode translation caching.
