# Changelog

All notable changes to this project are documented in this file.

## [1.0.3] - 2026-06-30

### Fixed
- Translation was silently blocked by CORS for any provider that doesn't
  send its own permissive CORS headers — including the default **ALX**
  provider and **Anthropic**. The CORS proxy that was meant to handle this
  only takes effect on hosts the extension has host permission for, but the
  broad `optional_host_permissions` it relied on was never actually
  requested, so on a real install it was a no-op. (OpenAI and Gemini kept
  working only because they send their own CORS headers.) This regressed the
  default provider in 1.0.2, which removed ALX's baked-in host permission.
- Options: switching provider now clears the API key field. A key is
  provider-specific, so the previous provider's key no longer carries over.
- Options: for the custom "OpenAI-compatible" provider, the API key label
  now notes that any value works if the server doesn't require one — the
  field stays required, so keyless local servers no longer leave the user
  guessing what to enter.

### Changed
- The built-in provider endpoints (Anthropic, OpenAI, Gemini, ALX) are now
  declared in `host_permissions`, so they work out of the box. A host
  permission alone makes the background fetch CORS-exempt.
- Custom ("OpenAI-compatible") endpoints now request host access for the
  host entered, at the moment the user saves — a scoped prompt instead of a
  silent CORS failure. Both `http` and `https` hosts are supported, so
  locally hosted models (e.g. `http://localhost:8080`) work too.

### Removed
- The CORS proxy and the permissions that only existed to power it:
  `declarativeNetRequest` (Chrome) and `webRequestBlocking` (Firefox). Header
  rewriting is no longer needed now that endpoint access comes from host
  permissions. Reduces the extension's footprint and review surface.
- `optional_host_permissions` is no longer relied on implicitly: the broad
  pattern is only ever requested one host at a time, at save, for a custom
  endpoint.

## [1.0.2] - 2026-06-30

### Changed
- Removed the `https://inference.alexandra.dk/*` host permission from both
  manifests. The Alexandra inference endpoint is reached through the
  user-granted `optional_host_permissions` like any other provider, so the
  baked-in host permission was unnecessary.

## [1.0.1] - 2026-06-27

### Added
- "ALX" provider preset (`inference.alexandra.dk`, `qwen3.5-397b`), now
  the default provider for new installs

### Fixed
- Chrome: extension hung on "Preparing English subtitles…". The service
  worker crashed on startup because it touched `chrome.webRequest`
  (unavailable on Chrome, which uses `declarativeNetRequest`) before
  registering the connection listener. APIs are now feature-detected.
- Chrome: page-world subtitle sniffer was blocked by DR's CSP. The
  fetch/XHR patch now runs as a dedicated `world: "MAIN"` content script
  instead of an inline-injected `<script>`.
- Chrome: registered the `subtitle-fetcher` content script (was
  Firefox-only) and route the master-manifest fetch through the page
  context, so subtitle data is no longer blocked by missing CDN host
  permissions.
- Options: switching provider now resets the endpoint and model to that
  provider's defaults instead of keeping the previous values.

## [1.0.0] - 2026-06-06

### Added
- Manifest V3 browser extension for Chrome and Firefox
- English subtitle injection for DR TV player
- Three-way subtitle toggle (Off / Dansk / English)
- LLM translation with user-configured API key (OpenAI, Anthropic, Gemini, custom)
- Per-episode translation cache in IndexedDB
- Options page for provider selection and API key management
- Privacy policy hosted on GitHub Pages
- Firefox add-on submission ready with `data_collection_permissions`
- Chrome Web Store submission ready
- Icons (16, 32, 48, 128px) and extension branding
- Build script (`build.mjs`) with esbuild bundling
- Source package for Mozilla review

### Changed
- Consolidated all release artifacts into `package/` directory
- Updated `PACKAGE_RELEASE_PLAN.md` with submission instructions

### Technical
- Background service worker with VTT sniffer and parser
- Content script with menu injection and track management
- CRLF normalization in VTT parsing
- Source maps included for review
