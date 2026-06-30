# DRTV in English — Source Code Submission

This document provides the build instructions and environment requirements for reproducing the extension build.

## Operating System Requirements

This extension can be built on any of the following:

- **macOS** 12+ (tested on macOS 14)
- **Linux** any modern distribution with Node.js support
- **Windows** 10+ with Node.js installed

## Build Environment Requirements

### Required Software

| Software | Version | Installation |
|----------|---------|--------------|
| Node.js | v20.x or v22.x or v24.x | https://nodejs.org/ |
| npm | v9.x or v10.x (bundled with Node) | Bundled with Node.js |

### Installation Instructions

1. **Install Node.js** (includes npm):
   - Download from https://nodejs.org/
   - Choose the LTS version (v20.x or v22.x) or Current (v24.x)
   - Follow the installer for your operating system

2. **Verify installation**:
   ```bash
   node --version
   npm --version
   ```

## Build Instructions

### Step 1: Install Dependencies

From the root directory of the source package:

```bash
npm install
```

This installs:
- `esbuild` v0.24.x — JavaScript bundler (bundles TypeScript → JavaScript; no minification)
- `sharp` v0.34.x — Image processing for icon generation
- `typescript` v5.5.x — Type checking
- `@types/chrome` — Chrome extension type definitions

### Step 2: Build the Extension

```bash
npm run build
```

This produces:
- `dist/chrome/` — Chrome Web Store version
- `dist/firefox/` — Firefox Add-ons version

Output includes:
- Bundled (non-minified) JavaScript files (`.js`)
- Source maps (`.js.map`) for debugging/review
- Static assets (HTML, icons, manifest)

### Step 3: (Optional) Package for Store Submission

```bash
npm run package
```

This creates:
- `dist/drtv-in-english-chrome-<version>.zip`
- `dist/drtv-in-english-firefox-<version>.zip`
- `dist/drtv-in-english-source-<version>.zip` (raw source for AMO review)

### Step 4: (Optional) Watch Mode for Development

```bash
npm run watch
```

Rebuilds automatically when source files change.

### Step 5: (Optional) Type Check

```bash
npm run typecheck
```

Runs TypeScript type checking without emitting files.

## Build Script

The build script is `build.mjs` in the root directory. It:

1. **Bundles TypeScript** → JavaScript using esbuild (no minification)
   - `src/background/index.ts` → `background/index.js`
   - `src/content/index.ts` → `content/index.js`
   - `src/content/early.ts` → `content/early.js`
   - `src/content/inject.ts` → `content/inject.js`
   - `src/content/subtitle-fetcher.ts` → `content/subtitle-fetcher.js`
   - `src/options/options.ts` → `options/options.js`

2. **Copies static files**:
   - `options/index.html` → `options/index.html`
   - Icons from `icons/` → `icons/`
   - Manifest from `manifest.chrome.json` or `manifest.firefox.json`

3. **Generates icons** (if needed):
   - Renders SVG source to PNG at 16, 32, 48, 128 pixels using Sharp

4. **Produces source maps**:
   - `.js.map` files alongside each bundled `.js` file

## Source Files

All source files are in the `src/` directory:

```
src/
  background/           # Service worker (VTT sniffer, parser, translator)
  content/              # Content scripts (menu injector, track manager)
  options/              # Options page (provider/API key form)
  shared/               # Shared types and utilities
```

**No transpiled/concatenated/minified files in source** — all source files are raw TypeScript/JavaScript, HTML, and JSON. Machine-generated output is only in `dist/` and should be ignored during review.

## Manifest Files

- `manifest.chrome.json` — Chrome Web Store manifest (MV3)
- `manifest.firefox.json` — Firefox Add-ons manifest (MV3 with `browser_specific_settings`)

## Additional Notes

- The extension uses **Manifest V3** for both Chrome and Firefox
- No external dependencies are fetched at runtime — all code is bundled
- API keys are stored in `chrome.storage.local` / `browser.storage.local`
- No telemetry, analytics, or remote data collection

## Contact

For questions about building or reviewing this extension:

- Developer: Dan Saattrup Smart
- Email: saattrupdan@gmail.com
- GitHub: https://github.com/saattrupdan/drtv_in_english
