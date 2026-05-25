// Phase 0 spike — DRTV in English
//
// Goals (per docs/extension-plan.md §Phase 0):
//   1. Find DR's <video> element on a DRTV episode page.
//   2. Attach a native TextTrack and add one hardcoded English VTTCue
//      covering 0–60s.
//   3. Hook DR's existing subtitle button into a 3-way Off/Dansk/English
//      menu. If that fails, fall back to a sibling "EN" pill.
//   4. Handle SPA navigations within DRTV (URL changes without reload).
//
// All logs are prefixed [drtv-en-spike].

(() => {
  const TAG = "[drtv-en-spike]";
  const SPIKE_TEXT = "★ Phase 0 spike — English cue 0–60s";
  const CUE_START = 0;
  const CUE_END = 60;
  const MARK = "data-drtv-en-spike";

  console.log(TAG, "content script loaded on", location.href);

  let currentVideo = null;
  let currentTrack = null;

  function waitForVideo(timeoutMs = 30000) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector("video");
      if (existing) return resolve(existing);

      const obs = new MutationObserver(() => {
        const v = document.querySelector("video");
        if (v) {
          obs.disconnect();
          resolve(v);
        }
      });
      obs.observe(document.documentElement, { childList: true, subtree: true });

      setTimeout(() => {
        obs.disconnect();
        reject(new Error("timed out waiting for <video>"));
      }, timeoutMs);
    });
  }

  function attachNativeTrack(video) {
    // Don't double-add if we already attached on this video element.
    for (const t of video.textTracks) {
      if (t.label === "English (spike)") {
        t.mode = "showing";
        return t;
      }
    }
    const track = video.addTextTrack("subtitles", "English (spike)", "en");
    track.mode = "showing";
    track.addCue(new VTTCue(CUE_START, CUE_END, SPIKE_TEXT));
    console.log(TAG, "native TextTrack attached", {
      trackCount: video.textTracks.length,
    });
    return track;
  }

  function findSubtitleButton() {
    // DR's player is Shaka-based; the subtitle button is a real
    // <button> in the light DOM (confirmed in the first spike run).
    // Cast a wide net by ARIA / class / data attrs.
    const all = document.querySelectorAll("button, [role='button']");
    const candidates = [];
    for (const el of all) {
      const aria = (el.getAttribute("aria-label") || "").toLowerCase();
      const cls = (el.className && el.className.toString
        ? el.className.toString()
        : ""
      ).toLowerCase();
      const dataAttrs = Array.from(el.attributes)
        .filter((a) => a.name.startsWith("data-"))
        .map((a) => `${a.name}=${a.value}`)
        .join(" ");
      const hay = `${aria} ${cls} ${dataAttrs}`.toLowerCase();
      if (/subtitle|caption|undertekst|cc\b|sprog|language/.test(hay)) {
        candidates.push(el);
      }
    }
    return candidates;
  }

  // ---- 3-way menu UI ---------------------------------------------------

  function ensureMenuStyles() {
    if (document.getElementById("drtv-en-spike-style")) return;
    const style = document.createElement("style");
    style.id = "drtv-en-spike-style";
    style.textContent = `
      .drtv-en-spike-menu {
        position: absolute;
        z-index: 2147483647;
        min-width: 140px;
        padding: 4px 0;
        background: rgba(20,20,20,0.95);
        color: white;
        border-radius: 6px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        font: 500 14px system-ui, -apple-system, sans-serif;
      }
      .drtv-en-spike-menu button {
        display: block;
        width: 100%;
        padding: 8px 14px;
        background: transparent;
        color: inherit;
        border: 0;
        text-align: left;
        cursor: pointer;
        font: inherit;
      }
      .drtv-en-spike-menu button:hover {
        background: rgba(255,255,255,0.12);
      }
      .drtv-en-spike-menu button[aria-checked="true"]::before {
        content: "✓ ";
      }
    `;
    document.head.appendChild(style);
  }

  let openMenu = null;
  let currentSelection = "off"; // off | dansk | english

  function closeMenu() {
    if (openMenu) {
      openMenu.remove();
      openMenu = null;
    }
  }

  function openMenuNear(button) {
    closeMenu();
    ensureMenuStyles();
    const menu = document.createElement("div");
    menu.className = "drtv-en-spike-menu";
    menu.setAttribute(MARK, "menu");

    const opts = [
      ["off", "Off"],
      ["dansk", "Dansk"],
      ["english", "English"],
    ];
    for (const [key, label] of opts) {
      const item = document.createElement("button");
      item.type = "button";
      item.textContent = label;
      item.setAttribute("aria-checked", String(key === currentSelection));
      item.addEventListener("click", (e) => {
        e.stopPropagation();
        currentSelection = key;
        console.log(TAG, "selection:", key);
        // Spike behavior: just toggle our English track visibility.
        if (currentTrack) {
          currentTrack.mode = key === "english" ? "showing" : "hidden";
        }
        closeMenu();
      });
      menu.appendChild(item);
    }

    // Position above the button.
    const r = button.getBoundingClientRect();
    menu.style.left = `${window.scrollX + r.left}px`;
    menu.style.top = `${window.scrollY + r.top - 8}px`;
    menu.style.transform = "translateY(-100%)";
    document.body.appendChild(menu);
    openMenu = menu;

    // Click-away.
    setTimeout(() => {
      const away = (e) => {
        if (!menu.contains(e.target)) {
          closeMenu();
          document.removeEventListener("click", away, true);
        }
      };
      document.addEventListener("click", away, true);
    }, 0);
  }

  function hookExistingButton(button) {
    if (button.hasAttribute(MARK)) return true;
    button.setAttribute(MARK, "hooked");
    // Capture-phase listener so we intercept before DR's own handler.
    button.addEventListener(
      "click",
      (e) => {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        openMenuNear(button);
      },
      true
    );
    console.log(TAG, "hooked existing subtitle button", button);
    return true;
  }

  function insertSiblingPill(reference) {
    if (document.querySelector(`[${MARK}="pill"]`)) return;
    const pill = document.createElement("button");
    pill.setAttribute(MARK, "pill");
    pill.textContent = "EN";
    pill.title = "DRTV in English (spike)";
    Object.assign(pill.style, {
      marginLeft: "8px",
      padding: "2px 8px",
      background: "#0a84ff",
      color: "white",
      border: "0",
      borderRadius: "999px",
      font: "600 12px system-ui, sans-serif",
      cursor: "pointer",
      zIndex: "2147483647",
    });
    pill.addEventListener("click", (e) => {
      e.stopPropagation();
      openMenuNear(pill);
    });
    reference.insertAdjacentElement("afterend", pill);
    console.log(TAG, "inserted fallback EN pill next to", reference);
  }

  function attachControl() {
    const candidates = findSubtitleButton();
    console.log(TAG, `subtitle-button candidates: ${candidates.length}`);
    if (!candidates.length) return false;
    const first = candidates[0];
    if (first.hasAttribute(MARK)) return true;
    // Primary path: extend the existing button into our 3-way menu.
    return hookExistingButton(first);
  }

  // ---- Lifecycle -------------------------------------------------------

  async function run() {
    let video;
    try {
      video = await waitForVideo();
    } catch (err) {
      console.error(TAG, err);
      return;
    }
    if (video === currentVideo) return;
    currentVideo = video;
    console.log(TAG, "found <video>", video);

    try {
      currentTrack = attachNativeTrack(video);
    } catch (err) {
      console.error(TAG, "addTextTrack failed", err);
    }

    // The player controls often mount lazily. Try a few times, then
    // also watch the DOM until the button shows up.
    let hooked = attachControl();
    if (!hooked) {
      const obs = new MutationObserver(() => {
        if (attachControl()) obs.disconnect();
      });
      obs.observe(document.body, { childList: true, subtree: true });
      // Give up after 20s and fall back to the sibling pill anchored
      // on whatever button looks plausible — or skip if nothing fits.
      setTimeout(() => {
        obs.disconnect();
        if (document.querySelector(`[${MARK}="hooked"]`)) return;
        const c = findSubtitleButton();
        if (c.length) insertSiblingPill(c[0]);
      }, 20000);
    }
  }

  // SPA navigation: DRTV changes URL without a full reload. Patch
  // history methods and listen for popstate; re-run when the URL
  // matches an episode path. Proper handling is Phase 3; this is just
  // enough to make the spike usable.
  function onUrlChange() {
    const m = /\/drtv\/(se|episode)\//.test(location.pathname);
    console.log(TAG, "url change", location.href, "match:", m);
    if (!m) return;
    // New page — reset and re-run.
    currentVideo = null;
    currentTrack = null;
    closeMenu();
    // Remove our hooks so we re-attach on the new player.
    document
      .querySelectorAll(`[${MARK}]`)
      .forEach((el) => el.removeAttribute(MARK));
    run();
  }

  for (const m of ["pushState", "replaceState"]) {
    const orig = history[m];
    history[m] = function (...args) {
      const ret = orig.apply(this, args);
      window.dispatchEvent(new Event("drtv-en-spike-locationchange"));
      return ret;
    };
  }
  window.addEventListener("popstate", onUrlChange);
  window.addEventListener("drtv-en-spike-locationchange", onUrlChange);

  run();
})();
