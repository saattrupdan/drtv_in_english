# Phase 0 spike — load and verify

Per `docs/extension-plan.md` §Phase 0. Goal: prove we can render an English
cue over DR's actual player, and that we have a path to a clickable in-player
control. Test episode:
`https://www.dr.dk/drtv/se/tingbjerg_eksperimentet_-de-udvalgte_594476`.

## Chrome

1. `chrome://extensions` → toggle **Developer mode** (top right).
2. **Load unpacked** → select `extension/spike/`.
3. Open the test episode in a new tab.
4. Open DevTools console and filter on `[drtv-en-spike]`.
5. Press play. Expected:
   - One log line `found <video>` followed by `native TextTrack attached`
     and `overlay attached`.
   - The text **★ Phase 0 spike — English cue 0–60s** visible over the
     player for the first 60 seconds (overlay path will always show it;
     the native track may or may not, depending on whether DR masks
     native cues).
   - One or more `subtitle-button candidates` log lines, ideally with
     a blue **EN** pill rendered next to DR's existing subtitle button.
6. Toggle fullscreen — the overlay should still appear. The native
   track depends on the browser's own renderer.

## Firefox

1. `about:debugging#/runtime/this-firefox` → **Load Temporary Add-on**.
2. Pick `extension/spike/manifest.json`.
3. Same verification steps as Chrome. Temporary add-ons unload on
   browser restart — that's fine for the spike.

## What we are looking to learn

| Question                                                  | Where to look                         |
| --------------------------------------------------------- | ------------------------------------- |
| Does `addTextTrack` + `VTTCue` render through DR's UI?    | Watch player, compare to overlay      |
| Does the cue survive fullscreen?                          | Toggle fullscreen, both paths         |
| Does DRM playback affect cue rendering?                   | This episode is Widevine — just play  |
| Can we find DR's subtitle button by ARIA / class?         | `subtitle-button candidates` log line |
| Can we drop a sibling widget next to it?                  | Blue **EN** pill appears in controls  |

If the native TextTrack is masked by DR's renderer, Phase 1+ relies on
the overlay path. If the sibling pill appears in a sensible place, the
"sibling pill fallback" path from the plan is viable; if not, we will
need to hook the existing button directly.

## Tearing down

- Chrome: `chrome://extensions` → **Remove**.
- Firefox: `about:debugging` → **Remove**, or restart Firefox.
