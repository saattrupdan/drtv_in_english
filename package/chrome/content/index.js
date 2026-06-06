"use strict";
(() => {
  // src/shared/messaging.ts
  function extractEpisodeId(url) {
    try {
      const u = new URL(url);
      if (!u.hostname.endsWith("dr.dk")) return null;
      const m = u.pathname.match(/\/drtv\/se\/[^/]*_([^/?#]+)/) ?? u.pathname.match(/\/drtv\/episode\/([^/?#]+)/);
      return m ? m[1] ?? null : null;
    } catch {
      return null;
    }
  }

  // src/shared/types.ts
  var PORT_NAME = "drtv-en";

  // src/content/menu.ts
  var MARK = "data-drtv-en";
  var BUTTON_RE = /subtitle|caption|undertekst|cc\b|sprog|language/;
  var bypassNext = false;
  function attachMenu(opts) {
    ensureStyles();
    return new Promise((resolve) => {
      let controller = null;
      let currentSelection = "off";
      let currentTop = opts.initialTopPosition;
      let drButton = null;
      const tryAttach = () => {
        const btn = findSubtitleButton();
        if (!btn) return false;
        if (btn.hasAttribute(MARK)) return true;
        btn.setAttribute(MARK, "hooked");
        drButton = btn;
        btn.addEventListener(
          "click",
          (e) => {
            if (bypassNext) return;
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            openMenu(
              btn,
              currentSelection,
              currentTop,
              (mode) => {
                currentSelection = mode;
                opts.onPick(mode);
              },
              (top) => {
                currentTop = top;
                opts.onTogglePosition(top);
              }
            );
          },
          true
        );
        console.log("[drtv-en/content] hooked DR subtitle button");
        controller = {
          setSelection(mode) {
            currentSelection = mode;
          },
          setTopPosition(top) {
            currentTop = top;
          },
          dispatchDrClick() {
            if (!drButton) return false;
            bypassNext = true;
            try {
              drButton.dispatchEvent(
                new MouseEvent("click", { bubbles: true, cancelable: true })
              );
            } finally {
              bypassNext = false;
            }
            return true;
          },
          destroy() {
            btn.removeAttribute(MARK);
          }
        };
        resolve(controller);
        return true;
      };
      if (tryAttach()) return;
      const obs = new MutationObserver(() => {
        if (tryAttach()) obs.disconnect();
      });
      obs.observe(document.body, { childList: true, subtree: true });
      setTimeout(() => {
        if (controller) return;
        obs.disconnect();
        const candidate = findSubtitleButton();
        if (!candidate) return;
        const pill = insertPill(
          candidate,
          (mode) => {
            currentSelection = mode;
            opts.onPick(mode);
          },
          (top) => {
            currentTop = top;
            opts.onTogglePosition(top);
          },
          () => currentSelection,
          () => currentTop
        );
        controller = {
          setSelection(mode) {
            currentSelection = mode;
          },
          setTopPosition(top) {
            currentTop = top;
          },
          dispatchDrClick() {
            return false;
          },
          destroy() {
            pill.remove();
          }
        };
        resolve(controller);
      }, 2e4);
    });
  }
  function findSubtitleButton() {
    const all = document.querySelectorAll("button, [role='button']");
    for (const el of all) {
      if (el.hasAttribute(MARK) && el.getAttribute(MARK) !== "hooked") continue;
      const aria = (el.getAttribute("aria-label") || "").toLowerCase();
      const rawCls = el.className;
      const cls = (typeof rawCls === "string" ? rawCls : String(rawCls ?? "")).toLowerCase();
      if (BUTTON_RE.test(`${aria} ${cls}`)) return el;
    }
    return null;
  }
  var openEl = null;
  function closeMenu() {
    if (openEl) {
      openEl.remove();
      openEl = null;
    }
  }
  function openMenu(anchor, current, currentTop, onPick, onTogglePosition) {
    closeMenu();
    const menu = document.createElement("div");
    menu.className = "drtv-en-menu";
    menu.setAttribute(MARK, "menu");
    for (const [key, label] of [
      ["off", "Off"],
      ["dansk", "Dansk"],
      ["english", "English"]
    ]) {
      const item = document.createElement("button");
      item.type = "button";
      item.textContent = label;
      item.setAttribute("aria-checked", String(key === current));
      item.addEventListener("click", (e) => {
        e.stopPropagation();
        onPick(key);
        closeMenu();
      });
      menu.appendChild(item);
    }
    const sep = document.createElement("div");
    sep.className = "drtv-en-sep";
    menu.appendChild(sep);
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "drtv-en-toggle";
    toggle.textContent = currentTop ? "Move subs to bottom" : "Move subs to top";
    toggle.title = "Use this if Danish subtitles are baked into the video and overlap with the English ones";
    toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      onTogglePosition(!currentTop);
      closeMenu();
    });
    menu.appendChild(toggle);
    const r = anchor.getBoundingClientRect();
    menu.style.left = `${window.scrollX + r.left}px`;
    menu.style.top = `${window.scrollY + r.top - 8}px`;
    menu.style.transform = "translateY(-100%)";
    document.body.appendChild(menu);
    openEl = menu;
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
  function insertPill(reference, onPick, onTogglePosition, getCurrent, getCurrentTop) {
    const pill = document.createElement("button");
    pill.setAttribute(MARK, "pill");
    pill.textContent = "EN";
    pill.title = "DRTV in English";
    Object.assign(pill.style, {
      marginLeft: "8px",
      padding: "2px 8px",
      background: "#0a84ff",
      color: "white",
      border: "0",
      borderRadius: "999px",
      font: "600 12px system-ui, sans-serif",
      cursor: "pointer"
    });
    pill.addEventListener("click", (e) => {
      e.stopPropagation();
      openMenu(pill, getCurrent(), getCurrentTop(), onPick, onTogglePosition);
    });
    reference.insertAdjacentElement("afterend", pill);
    return pill;
  }
  function ensureStyles() {
    if (document.getElementById("drtv-en-style")) return;
    const style = document.createElement("style");
    style.id = "drtv-en-style";
    style.textContent = `
    .drtv-en-menu {
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
    .drtv-en-menu button {
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
    .drtv-en-menu button:hover {
      background: rgba(255,255,255,0.12);
    }
    .drtv-en-menu button[aria-checked="true"]::before {
      content: "\u2713 ";
    }
    .drtv-en-menu .drtv-en-sep {
      height: 1px;
      margin: 4px 0;
      background: rgba(255,255,255,0.15);
    }
    .drtv-en-menu button.drtv-en-toggle {
      font-size: 12px;
      opacity: 0.85;
    }
    .drtv-en-status {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      z-index: 2147483647;
      padding: 14px 20px;
      background: rgba(20,20,20,0.82);
      color: white;
      font: 500 14px system-ui, -apple-system, sans-serif;
      border-radius: 10px;
      pointer-events: none;
      display: flex;
      align-items: center;
      gap: 12px;
      box-shadow: 0 6px 24px rgba(0,0,0,0.4);
    }
    .drtv-en-status .drtv-en-spinner {
      display: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      border: 2px solid rgba(255,255,255,0.25);
      border-top-color: #fff;
      animation: drtv-en-spin 0.8s linear infinite;
    }
    .drtv-en-status.drtv-en-busy .drtv-en-spinner {
      display: inline-block;
    }
    @keyframes drtv-en-spin {
      to { transform: rotate(360deg); }
    }
  `;
    document.head.appendChild(style);
  }

  // src/content/track-injector.ts
  var LABEL = "English (DRTV in English)";
  var MARK2 = "data-drtv-en-track";
  var TrackManager = class _TrackManager {
    video;
    track = null;
    // When true, render cues near the top of the frame — used to dodge
    // burnt-in Danish subtitles on the accessibility ("tale-tekstning")
    // variants of some DR shows.
    topPosition = false;
    constructor(video) {
      this.video = video;
    }
    // Vertical inset (percent of player height) from the nearest edge, so
    // our overlay sits in the same band DR uses for its own Danish subs
    // instead of flush against the frame edge.
    static EDGE_GAP_PCT = 12;
    setTopPosition(top) {
      if (this.topPosition === top) return;
      this.topPosition = top;
      const cues = this.track?.cues;
      if (!cues) return;
      for (let i = 0; i < cues.length; i++) {
        const cue = cues[i];
        if (cue instanceof VTTCue) this.positionCue(cue);
      }
    }
    positionCue(cue) {
      cue.snapToLines = false;
      cue.line = this.topPosition ? _TrackManager.EDGE_GAP_PCT : 100 - _TrackManager.EDGE_GAP_PCT;
    }
    ensureTrack() {
      if (this.track) return this.track;
      for (const t of Array.from(this.video.textTracks)) {
        if (t.label === LABEL) {
          this.track = t;
          return t;
        }
      }
      const track = this.video.addTextTrack("subtitles", LABEL, "en");
      this.video.setAttribute(MARK2, "1");
      this.track = track;
      return track;
    }
    addCues(cues) {
      const track = this.ensureTrack();
      const existing = /* @__PURE__ */ new Set();
      const trackCues = track.cues;
      if (trackCues) {
        for (let i = 0; i < trackCues.length; i++) {
          const c = trackCues[i];
          existing.add(`${c.startTime.toFixed(3)}|${c.endTime.toFixed(3)}|${c.text}`);
        }
      }
      for (const c of cues) {
        const key = `${c.start.toFixed(3)}|${c.end.toFixed(3)}|${c.text}`;
        if (existing.has(key)) continue;
        try {
          const vtt = new VTTCue(c.start, c.end, c.text);
          this.positionCue(vtt);
          track.addCue(vtt);
        } catch (err) {
          console.warn("[drtv-en/content] addCue failed", err, c);
        }
      }
    }
    clear() {
      if (!this.track) return;
      const cues = this.track.cues;
      if (!cues) return;
      for (let i = cues.length - 1; i >= 0; i--) {
        const cue = cues[i];
        if (cue) this.track.removeCue(cue);
      }
    }
    applyMode(mode) {
      const track = this.ensureTrack();
      track.mode = mode === "english" ? "showing" : "hidden";
    }
  };

  // src/content/buffer-guard.ts
  var LOOKAHEAD_S = 0.5;
  var RUNWAY_BASE_S = 2;
  var RUNWAY_MAX_S = 30;
  var RUNWAY_STREAK_RESET_MS = 15e3;
  var BufferGuard = class {
    video;
    onStatus;
    starts = [];
    ready = [];
    active = false;
    autoPaused = false;
    userPaused = false;
    requiredRunwayS = RUNWAY_BASE_S;
    lastResumeAt = 0;
    constructor(video, onStatus) {
      this.video = video;
      this.onStatus = onStatus;
      video.addEventListener("timeupdate", this.check);
      video.addEventListener("seeking", this.check);
      video.addEventListener("pause", this.onPause);
      video.addEventListener("play", this.onPlay);
    }
    setSchedule(starts) {
      this.starts = starts;
      this.ready = new Array(starts.length).fill(false);
      this.check();
    }
    // Gate playback immediately, before any schedule arrives. Called when
    // the user picks English so the video pauses right away instead of
    // playing on for the second-or-two it takes the background to fetch
    // and parse the Danish VTT.
    beginGate() {
      this.active = true;
      this.userPaused = false;
      this.autoPaused = true;
      this.requiredRunwayS = RUNWAY_BASE_S;
      if (!this.video.paused) this.video.pause();
      this.onStatus("Preparing English subtitles\u2026");
    }
    markReady(cueStarts) {
      for (const t of cueStarts) {
        const idx = this.findStart(t);
        if (idx >= 0) this.ready[idx] = true;
      }
      this.check();
    }
    reset() {
      this.starts = [];
      this.ready = [];
      if (this.autoPaused) this.resume();
      this.onStatus(null);
    }
    enable() {
      this.active = true;
      this.check();
    }
    disable() {
      this.active = false;
      if (this.autoPaused) this.resume();
      this.onStatus(null);
    }
    // Stop the guard without clearing the overlay — used when the
    // translation is done and we want the "English subtitles ready"
    // message to persist without check() overwriting it.
    stop() {
      this.active = false;
      this.video.removeEventListener("timeupdate", this.check);
      this.video.removeEventListener("seeking", this.check);
    }
    destroy() {
      this.video.removeEventListener("timeupdate", this.check);
      this.video.removeEventListener("seeking", this.check);
      this.video.removeEventListener("pause", this.onPause);
      this.video.removeEventListener("play", this.onPlay);
    }
    findStart(t) {
      for (let i = 0; i < this.starts.length; i++) {
        if (Math.abs(this.starts[i] - t) < 1e-3) return i;
      }
      return -1;
    }
    check = () => {
      if (!this.active) return;
      if (this.starts.length === 0) {
        if (this.autoPaused) {
          if (!this.video.paused) this.video.pause();
          this.onStatus(this.bufferLabel(0));
        }
        return;
      }
      const t = this.video.currentTime;
      const frontier = this.frontierAhead(t);
      if (frontier === -1) {
        if (this.autoPaused) this.resume();
        this.onStatus(null);
        return;
      }
      const frontierTime = this.starts[frontier];
      const gap = frontierTime - t;
      const nextIdx = this.firstAhead(t);
      const nextReady = nextIdx >= 0 && this.ready[nextIdx];
      if (this.autoPaused) {
        if (nextReady && gap >= this.requiredRunwayS) {
          this.resume();
          this.onStatus(null);
        } else {
          this.onStatus(this.bufferLabel(gap));
        }
        return;
      }
      if (!nextReady || gap < LOOKAHEAD_S) {
        if (this.lastResumeAt > 0 && Date.now() - this.lastResumeAt < RUNWAY_STREAK_RESET_MS) {
          this.requiredRunwayS = Math.min(
            RUNWAY_MAX_S,
            this.requiredRunwayS * 2
          );
        } else {
          this.requiredRunwayS = RUNWAY_BASE_S;
        }
        this.onStatus(this.bufferLabel(gap));
        if (!this.video.paused) {
          this.autoPaused = true;
          this.video.pause();
        }
      } else {
        this.onStatus(null);
      }
    };
    bufferLabel(_gap) {
      return "Buffering English subtitles\u2026";
    }
    frontierAhead(t) {
      for (let i = 0; i < this.starts.length; i++) {
        if (this.starts[i] < t - 0.25) continue;
        if (!this.ready[i]) return i;
      }
      return -1;
    }
    firstAhead(t) {
      for (let i = 0; i < this.starts.length; i++) {
        if (this.starts[i] >= t - 0.25) return i;
      }
      return -1;
    }
    resume() {
      this.autoPaused = false;
      this.lastResumeAt = Date.now();
      if (this.userPaused) return;
      void this.video.play().catch(() => {
      });
    }
    onPause = () => {
      if (!this.autoPaused) this.userPaused = true;
    };
    onPlay = () => {
      this.userPaused = false;
      if (this.active && this.autoPaused) {
        setTimeout(() => {
          if (this.autoPaused && !this.video.paused) {
            this.video.pause();
            this.onStatus(this.bufferLabel(0));
          }
        }, 0);
      }
    };
  };

  // src/content/index.ts
  var TAG = "[drtv-en/content]";
  var STALL_HINT_MS = 6e4;
  var state = null;
  var statusEl = null;
  var statusTextEl = null;
  async function bootEpisode(episodeId) {
    const video = await waitForVideo();
    if (!video) {
      console.warn(TAG, "no <video> appeared in time");
      return;
    }
    if (state && state.episodeId !== episodeId) return;
    const track = new TrackManager(video);
    track.ensureTrack();
    const guard = new BufferGuard(video, (msg) => {
      if (msg) showStatus(msg, { busy: true });
      else hideStatus();
    });
    attachStatusOverlay(video);
    let lastSent = -1;
    const onSeek = () => {
      const t = Math.floor(video.currentTime);
      if (t === lastSent) return;
      lastSent = t;
      sendForState(s, { type: "seek", time: video.currentTime });
    };
    video.addEventListener("seeking", onSeek);
    const port = chrome.runtime.connect({ name: PORT_NAME });
    port.onMessage.addListener(onPortEvent);
    const s = {
      episodeId,
      video,
      track,
      menu: null,
      // attached just below
      guard,
      port,
      drSubsOn: false,
      destroyed: false,
      stallTimer: null,
      lastProgressAt: 0
    };
    state = s;
    port.onDisconnect.addListener(() => {
      console.log(TAG, "port disconnected");
      if (state === s) {
        s.port = null;
      }
    });
    sendForState(s, {
      type: "episode-active",
      episodeId,
      url: location.href
    });
    s.menu = await attachMenu({
      initialTopPosition: false,
      onPick: (mode) => {
        if (state === s && !s.destroyed) applySelection(s, mode);
      },
      onTogglePosition: (top) => {
        if (state !== s || s.destroyed) return;
        s.track.setTopPosition(top);
        s.menu.setTopPosition(top);
      }
    });
    if (s.destroyed) return;
    console.log(TAG, "ready for episode", episodeId);
  }
  function teardownEpisode(reason) {
    if (!state) return;
    const s = state;
    s.destroyed = true;
    console.log(TAG, "teardown:", reason, s.episodeId);
    try {
      s.port?.disconnect();
    } catch {
    }
    s.guard.destroy();
    s.track.clear();
    s.menu.destroy();
    stopStallWatchdog(s);
    hideStatus();
    state = null;
  }
  function startStallWatchdog(s) {
    stopStallWatchdog(s);
    s.lastProgressAt = Date.now();
    s.stallTimer = setInterval(() => {
      if (s.destroyed) return;
      if (Date.now() - s.lastProgressAt < STALL_HINT_MS) return;
      showStatus(
        "Still translating \u2014 provider seems slow. Pick Off and English again to retry.",
        { busy: true }
      );
    }, 5e3);
  }
  function stopStallWatchdog(s) {
    if (s.stallTimer !== null) {
      clearInterval(s.stallTimer);
      s.stallTimer = null;
    }
  }
  function bumpStallWatchdog(s) {
    s.lastProgressAt = Date.now();
  }
  function applySelection(s, mode) {
    console.log(TAG, "selection:", mode);
    s.track.applyMode(mode);
    s.menu.setSelection(mode);
    const wantDrOn = mode === "dansk";
    if (wantDrOn !== s.drSubsOn) {
      if (s.menu.dispatchDrClick()) s.drSubsOn = wantDrOn;
    }
    if (mode === "english") {
      if (!hasEnglishCues(s.track)) {
        s.guard.reset();
        s.guard.beginGate();
        startStallWatchdog(s);
        sendForState(s, {
          type: "request-translate",
          episodeId: s.episodeId,
          playhead: s.video.currentTime
        });
      } else {
        s.guard.enable();
      }
    } else {
      s.guard.disable();
      sendForState(s, { type: "cancel" });
      stopStallWatchdog(s);
      hideStatus();
    }
  }
  function hasEnglishCues(t) {
    return (t.ensureTrack().cues?.length ?? 0) > 0;
  }
  function sendForState(s, msg) {
    try {
      s.port?.postMessage(msg);
    } catch (err) {
      console.warn(TAG, "port send failed", err);
    }
  }
  function onPortEvent(event) {
    const s = state;
    if (!s) {
      console.warn(TAG, "port event dropped \u2014 no state", event.type);
      return;
    }
    console.log(TAG, "port event:", event.type);
    switch (event.type) {
      case "status":
        bumpStallWatchdog(s);
        showStatus(formatStatus(event.state), { busy: true });
        break;
      case "schedule":
        bumpStallWatchdog(s);
        s.guard.setSchedule(event.starts);
        break;
      case "cues":
        stopStallWatchdog(s);
        s.track.addCues(event.cues);
        break;
      case "done":
        stopStallWatchdog(s);
        s.guard.stop();
        showStatus("English subtitles ready", { hideAfterMs: 2500 });
        break;
      case "error":
        stopStallWatchdog(s);
        s.guard.disable();
        showStatus(event.message, { hideAfterMs: 8e3 });
        break;
      case "heartbeat":
        break;
    }
  }
  function formatStatus(state2) {
    switch (state2) {
      case "waiting-for-vtt":
        return "Waiting for DRTV subtitles\u2026";
      case "fetching-vtt":
        return "Fetching Danish subtitles\u2026";
      case "parsing":
        return "Reading subtitles\u2026";
      case "translating":
        return "Translating subtitles to English\u2026";
      default:
        return state2;
    }
  }
  function attachStatusOverlay(video) {
    if (statusEl && statusEl.isConnected && statusEl.parentElement) {
      if (statusEl.parentElement.contains(video)) return;
      console.log(TAG, "re-hosting overlay \u2014 video moved");
      statusEl.remove();
      statusEl = null;
      statusTextEl = null;
    }
    const host = video.parentElement ?? document.body;
    console.log(TAG, "attachStatusOverlay host:", host.tagName, "video parent:", video.parentElement?.tagName);
    if (getComputedStyle(host).position === "static") {
      host.style.position = "relative";
    }
    statusEl = document.createElement("div");
    statusEl.className = "drtv-en-status";
    statusEl.style.display = "none";
    const spinner = document.createElement("div");
    spinner.className = "drtv-en-spinner";
    statusTextEl = document.createElement("span");
    statusEl.append(spinner, statusTextEl);
    host.appendChild(statusEl);
    console.log(TAG, "statusEl created, isConnected:", statusEl.isConnected);
  }
  function showStatus(text, opts = {}) {
    if (!statusEl || !statusTextEl) {
      console.warn(TAG, "showStatus called but statusEl is null");
      return;
    }
    console.log(TAG, "showStatus:", text, "busy:", !!opts.busy);
    statusTextEl.textContent = text;
    statusEl.classList.toggle("drtv-en-busy", !!opts.busy);
    statusEl.style.display = "flex";
    if (opts.hideAfterMs !== void 0) {
      setTimeout(() => hideStatus(), opts.hideAfterMs);
    }
  }
  function hideStatus() {
    console.log(TAG, "hideStatus called");
    if (statusEl) statusEl.style.display = "none";
  }
  function waitForVideo(timeoutMs = 3e4) {
    return new Promise((resolve) => {
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
        resolve(null);
      }, timeoutMs);
    });
  }
  function watchUrl() {
    let lastUrl = location.href;
    setInterval(() => {
      if (location.href === lastUrl) return;
      lastUrl = location.href;
      const nextId = extractEpisodeId(location.href);
      const currentId = state?.episodeId ?? null;
      if (nextId === currentId) return;
      teardownEpisode("spa-nav");
      if (nextId) void bootEpisode(nextId);
    }, 500);
  }
  function start() {
    const id = extractEpisodeId(location.href);
    if (id) void bootEpisode(id);
    watchUrl();
  }
  start();
})();
//# sourceMappingURL=index.js.map
