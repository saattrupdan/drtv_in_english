# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-06-06

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
