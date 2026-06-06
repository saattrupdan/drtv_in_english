"use strict";
(() => {
  // src/shared/types.ts
  var PORT_NAME = "drtv-en";

  // src/background/vtt-sniffer.ts
  var byTab = /* @__PURE__ */ new Map();
  function ensure(tabId) {
    let s = byTab.get(tabId);
    if (!s) {
      s = { playlists: /* @__PURE__ */ new Set(), segments: /* @__PURE__ */ new Set() };
      byTab.set(tabId, s);
    }
    return s;
  }
  var SEGMENT_RE = /^(.*\/subtitles\/[^/]+\/)segment_\d+\.vtt(\?|#|$)/i;
  function setVttUrlForTab(tabId, url) {
    const s = ensure(tabId);
    if (/\/master[^/]*\.m3u8/i.test(url)) {
      s.masterUrl = url;
      return;
    }
    if (/\/subtitles\/[^?#]*\.m3u8/i.test(url)) {
      s.playlists.add(url);
      return;
    }
    s.segments.add(url);
    const m = SEGMENT_RE.exec(url);
    if (m) s.playlists.add(`${m[1]}playlist.m3u8`);
  }
  function deriveMasterUrl(subs) {
    if (subs.masterUrl) return subs.masterUrl;
    const sample = subs.playlists.values().next().value ?? subs.segments.values().next().value;
    if (!sample) return void 0;
    const m = /^(.*)\/subtitles\//i.exec(sample);
    if (!m) return void 0;
    return `${m[1]}/stream_fmp4/master_manifest.m3u8`;
  }
  function getSubsForTab(tabId) {
    return byTab.get(tabId);
  }
  function chooseBestPlaylist(subs) {
    for (const p of subs.playlists) {
      if (/HardOfHearing/i.test(p)) return p;
    }
    for (const p of subs.playlists) {
      if (!/\/Foreign-/i.test(p)) return p;
    }
    return void 0;
  }
  function clearVttForTab(tabId) {
    byTab.delete(tabId);
  }
  function installVttSniffer() {
    chrome.webRequest.onCompleted.addListener(
      (details) => {
        if (details.tabId < 0) return;
        const url = details.url;
        const looksLikeVtt = /\.vtt(\?|#|$)/i.test(url) || /\.webvtt(\?|#|$)/i.test(url) || /\/subtitles\/[^?#]*\.m3u8/i.test(url) || /\/master[^/]*\.m3u8/i.test(url);
        if (!looksLikeVtt) return;
        setVttUrlForTab(details.tabId, url);
        console.log("[drtv-en/bg] captured VTT (webRequest)", details.tabId, url);
      },
      { urls: ["<all_urls>"] }
    );
    chrome.tabs.onRemoved.addListener((tabId) => {
      byTab.delete(tabId);
    });
  }

  // src/background/cors-proxy.ts
  var CORS_HEADERS = [
    { name: "Access-Control-Allow-Origin", value: "*" },
    { name: "Access-Control-Allow-Methods", value: "GET, POST, OPTIONS" },
    { name: "Access-Control-Allow-Headers", value: "Authorization, Content-Type" }
  ];
  async function enableCorsProxy() {
    const isFirefox = typeof browser !== "undefined";
    if (isFirefox) {
      enableCorsProxyFirefox();
    } else {
      await enableCorsProxyChrome();
    }
  }
  function enableCorsProxyFirefox() {
    chrome.webRequest.onHeadersReceived.addListener(
      (details) => {
        if (!details.responseHeaders) return {};
        const filtered = details.responseHeaders.filter(
          (h) => !h.name?.toLowerCase().startsWith("access-control-")
        );
        filtered.push(...CORS_HEADERS);
        return { responseHeaders: filtered };
      },
      {
        urls: ["<all_urls>"],
        types: ["xmlhttprequest"]
      },
      ["blocking", "responseHeaders"]
    );
    console.log("[drtv-en/bg] CORS proxy enabled via webRequest (Firefox)");
  }
  async function enableCorsProxyChrome() {
    const rules = [
      {
        id: 1,
        priority: 1,
        action: {
          type: "modifyHeaders",
          responseHeaders: CORS_HEADERS.map((h) => ({
            header: h.name,
            operation: "set",
            value: h.value
          }))
        },
        condition: {
          urlFilter: "*://*/*",
          resourceTypes: ["xmlhttprequest"]
        }
      }
    ];
    try {
      const existing = await chrome.declarativeNetRequest.getDynamicRules();
      if (existing.length > 0) {
        await chrome.declarativeNetRequest.updateDynamicRules({
          removeRuleIds: existing.map((r) => r.id)
        });
      }
      await chrome.declarativeNetRequest.updateDynamicRules({ addRules: rules });
      console.log("[drtv-en/bg] CORS proxy enabled via declarativeNetRequest (Chrome)");
    } catch (err) {
      console.warn("[drtv-en/bg] declarativeNetRequest failed, falling back to no-op", err);
    }
  }

  // src/background/vtt-parser.ts
  var CUE = new RegExp(
    String.raw`(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})[^\n]*\n` + String.raw`((?:.+\n?)+?)(?=\n\s*\n|\n[^\n]*-->|$)`,
    "gm"
  );
  var TAG = /<[^>]+>/g;
  function parseVtt(raw) {
    const text = raw.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const out = [];
    for (const m of text.matchAll(CUE)) {
      const start = parseTs(m[1]);
      const end = parseTs(m[2]);
      const lines = (m[3] ?? "").split("\n").map((l) => l.trim()).filter(Boolean);
      const cleaned = lines.join(" ").replace(TAG, "").trim();
      if (!cleaned) continue;
      out.push({ start, end, text: cleaned });
    }
    return out;
  }
  function parseTs(s) {
    const [h, mi, rest] = s.split(":");
    const [sec, ms] = (rest ?? "0.000").split(".");
    return Number(h) * 3600 + Number(mi) * 60 + Number(sec) + Number(ms ?? "0") / 1e3;
  }

  // src/background/playlist.ts
  async function pageFetch(url) {
    return new Promise((resolve, reject) => {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tabId = tabs[0]?.id;
        if (!tabId) {
          reject(new Error("No active tab"));
          return;
        }
        chrome.tabs.sendMessage(
          tabId,
          { type: "fetch-url", url },
          (response) => {
            if (chrome.runtime.lastError) {
              reject(new Error(chrome.runtime.lastError.message));
              return;
            }
            if (response && typeof response === "object" && "error" in response) {
              reject(new Error(response.error));
              return;
            }
            if (typeof response === "string") {
              resolve(response);
            } else {
              reject(new Error("Invalid response from content script"));
            }
          }
        );
      });
    });
  }
  async function fetchCuesFromPlaylist(playlistUrl, _signal) {
    const text = await pageFetch(playlistUrl);
    const segmentUris = parseM3u8Segments(text);
    if (segmentUris.length === 0) throw new Error("playlist had no segments");
    const base = new URL(playlistUrl);
    const segmentUrls = segmentUris.map((u) => new URL(u, base).toString());
    const results = await Promise.all(
      segmentUrls.map((u) => pageFetch(u))
    );
    const merged = [];
    const seen = /* @__PURE__ */ new Set();
    for (const body of results) {
      for (const cue of parseVtt(body)) {
        const key = `${cue.start.toFixed(3)}|${cue.text}`;
        if (seen.has(key)) continue;
        seen.add(key);
        merged.push(cue);
      }
    }
    merged.sort((a, b) => a.start - b.start);
    return merged;
  }
  function parseM3u8Segments(body) {
    return body.split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
  }

  // src/background/master-manifest.ts
  function parseSubtitleTracks(masterText) {
    const tracks = [];
    for (const line of masterText.split(/\r?\n/)) {
      if (!line.startsWith("#EXT-X-MEDIA")) continue;
      const attrs = parseAttrs(line);
      if (attrs.TYPE !== "SUBTITLES") continue;
      tracks.push({
        uri: attrs.URI ?? "",
        name: attrs.NAME ?? "",
        language: attrs.LANGUAGE ?? "",
        isDefault: attrs.DEFAULT === "YES"
      });
    }
    return tracks;
  }
  function pickDanishTrack(tracks) {
    const da = tracks.filter((t) => /^da\b/i.test(t.language));
    if (da.length === 0) return void 0;
    const named = da.find(
      (t) => /dansk/i.test(t.name) && !/fremmed/i.test(t.name)
    );
    if (named) return named;
    const nonDefault = da.find((t) => !t.isDefault);
    if (nonDefault) return nonDefault;
    return da[0];
  }
  function parseAttrs(line) {
    const out = {};
    const colon = line.indexOf(":");
    const body = colon >= 0 ? line.slice(colon + 1) : line;
    const re = /([A-Z0-9-]+)=(?:"([^"]*)"|([^,]*))/g;
    let m;
    while ((m = re.exec(body)) !== null) {
      const key = m[1];
      out[key] = m[2] ?? m[3] ?? "";
    }
    return out;
  }

  // src/shared/storage.ts
  var PROVIDER_PRESETS = {
    anthropic: {
      label: "Anthropic",
      endpoint: "https://api.anthropic.com/v1/messages",
      model: "claude-haiku-4-5"
    },
    openai: {
      label: "OpenAI",
      endpoint: "https://api.openai.com/v1/responses",
      model: "gpt-5-mini"
    },
    gemini: {
      label: "Gemini",
      endpoint: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
      model: "gemini-3.5-flash"
    },
    "openai-compatible": {
      label: "OpenAI-compatible",
      endpoint: "",
      model: ""
    }
  };
  var DEFAULTS = {
    provider: "anthropic",
    endpoint: PROVIDER_PRESETS.anthropic.endpoint,
    apiKey: "",
    model: PROVIDER_PRESETS.anthropic.model,
    batchSize: 5,
    contextWindow: 6,
    maxParallel: 20
  };
  var KEY = "providerConfig";
  async function loadProviderConfig() {
    const got = await chrome.storage.local.get(KEY);
    return { ...DEFAULTS, ...got[KEY] ?? {} };
  }
  async function loadMaxParallel(provider, model) {
    const key = `maxParallel:${provider}:${model}`;
    const got = await chrome.storage.local.get(key);
    return got[key] ?? null;
  }
  async function saveMaxParallel(provider, model, value) {
    const key = `maxParallel:${provider}:${model}`;
    await chrome.storage.local.set({ [key]: value });
  }

  // src/background/translate-llm.ts
  var SYSTEM_PROMPT = "You are an expert translator and editor of Danish TV subtitles. The source text was transcribed by an ASR model and may contain errors (missing words, mangled proper nouns, literal translations of idioms). Your task is to: 1) Correct any ASR errors using the surrounding context. 2) Translate the corrected text into the requested target language. 3) Preserve the original meaning, speaker intent, and subtitle length. 4) Do not translate proper nouns (keep them as-is). 5) Return ONLY the requested JSON object, with no extra prose.";
  var PROVIDER_TIMEOUT_MS = 9e4;
  function withTimeout(signal, ms) {
    const ctrl = new AbortController();
    const onAbort = () => ctrl.abort(signal.reason);
    if (signal) {
      if (signal.aborted) ctrl.abort(signal.reason);
      else signal.addEventListener("abort", onAbort, { once: true });
    }
    const timer = setTimeout(
      () => ctrl.abort(new Error(`provider request timed out after ${ms}ms`)),
      ms
    );
    return {
      signal: ctrl.signal,
      cancel: () => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
      }
    };
  }
  async function translateWithLLM(cues, cfg, opts) {
    if (cues.length === 0) return 0;
    const batches = buildBatches(cues, cfg.batchSize, cfg.contextWindow);
    let emitted = 0;
    let failedCount = 0;
    let totalBatches = 0;
    const persistedMaxParallel = await loadMaxParallel(cfg.provider, cfg.model);
    let currentConcurrency = persistedMaxParallel ?? cfg.maxParallel;
    const attemptWithConcurrency = async (concurrency) => {
      const remaining = new Set(batches.map((_, i) => i));
      let roundFailures = 0;
      let roundTotal = 0;
      const pickNext = () => {
        if (remaining.size === 0) return void 0;
        const p = opts.getPlayhead?.() ?? 0;
        let bestI;
        let bestD = Infinity;
        for (const i of remaining) {
          const indices = batches[i].targetIndices;
          const firstCue = cues[indices[0]];
          const lastCue = cues[indices[indices.length - 1]];
          let d;
          if (firstCue.start <= p && p <= lastCue.end) {
            d = 0;
          } else if (firstCue.start >= p) {
            d = firstCue.start - p;
          } else {
            d = (p - firstCue.start) * 10 + 1e3;
          }
          if (d < bestD) {
            bestD = d;
            bestI = i;
          }
        }
        if (bestI === void 0) return void 0;
        remaining.delete(bestI);
        return bestI;
      };
      const workers = [];
      for (let w = 0; w < Math.min(concurrency, remaining.size); w++) {
        workers.push(
          (async () => {
            while (remaining.size > 0) {
              if (opts.signal?.aborted) return;
              const i = pickNext();
              if (i === void 0) return;
              const { targetIndices, userPrompt } = batches[i];
              roundTotal++;
              const translations = await runBatch(
                cfg,
                userPrompt,
                targetIndices,
                opts.signal
              ).catch((err) => {
                console.warn(
                  "[drtv-en/bg] batch failed",
                  targetIndices,
                  err
                );
                roundFailures++;
                return /* @__PURE__ */ new Map();
              });
              if (opts.signal?.aborted) return;
              const batchCues = targetIndices.map((idx) => {
                const src = cues[idx];
                const text = translations.get(idx)?.trim() || src.text;
                return { start: src.start, end: src.end, text };
              });
              emitted += batchCues.length;
              opts.onBatch(batchCues);
            }
          })()
        );
      }
      await Promise.all(workers);
      const failureRate = roundTotal > 0 ? roundFailures / roundTotal : 0;
      const success = failureRate <= 0.5;
      let saved = false;
      if (success && persistedMaxParallel === null) {
        await saveMaxParallel(cfg.provider, cfg.model, concurrency);
        saved = true;
      }
      return { success, saved };
    };
    while (currentConcurrency >= 1) {
      if (opts.signal?.aborted) break;
      const result = await attemptWithConcurrency(currentConcurrency);
      if (result.success) {
        return emitted;
      }
      currentConcurrency = Math.max(1, Math.floor(currentConcurrency / 2));
      console.log(
        "[drtv-en/bg] reducing concurrency to",
        currentConcurrency,
        "due to failures"
      );
    }
    return emitted;
  }
  function buildBatches(cues, batchSize, contextWindow) {
    const n = cues.length;
    const out = [];
    for (let start = 0; start < n; start += batchSize) {
      const end = Math.min(n, start + batchSize);
      const windowStart = Math.max(0, start - contextWindow);
      const windowEnd = Math.min(n, end + contextWindow);
      let numbered = "";
      for (let j = windowStart; j < windowEnd; j++) {
        const c = cues[j];
        const text = c.text || "[no text]";
        numbered += `${j}: [${c.start.toFixed(3)} -> ${c.end.toFixed(3)}] ${text}
`;
      }
      const targetIndices = [];
      for (let i = start; i < end; i++) targetIndices.push(i);
      const targetIds = targetIndices.join(", ");
      const userPrompt = `Translate the Danish subtitles to English. Output ONLY English text.

Chunks (translate ONLY the chunks with ids ${targetIds}):

${numbered}---
Return a JSON object of the form {"translations": {"<id>": "<corrected and translated text>", ...}} covering exactly the requested ids (${targetIds}).`;
      out.push({ targetIndices, userPrompt });
    }
    return out;
  }
  async function runBatch(cfg, userPrompt, targetIndices, signal) {
    const raw = await callProvider(cfg, userPrompt, signal);
    console.log("[drtv-en/bg] llm raw response (truncated):", raw?.slice(0, 500));
    const parsed = parseTranslations(raw, targetIndices);
    console.log("[drtv-en/bg] llm parsed:", parsed.size, "cues");
    return parsed;
  }
  async function callProvider(cfg, userPrompt, signal) {
    const { signal: boundedSignal, cancel } = withTimeout(
      signal,
      PROVIDER_TIMEOUT_MS
    );
    try {
      switch (cfg.provider) {
        case "anthropic":
          return await callAnthropic(cfg, userPrompt, boundedSignal);
        case "openai":
          return await callOpenAIResponses(cfg, userPrompt, boundedSignal);
        case "gemini":
        case "openai-compatible":
          return await callChatCompletions(cfg, userPrompt, boundedSignal);
        default:
          throw new Error(`unsupported provider: ${cfg.provider}`);
      }
    } finally {
      cancel();
    }
  }
  async function callAnthropic(cfg, userPrompt, signal) {
    const body = {
      model: cfg.model,
      max_tokens: 4096,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: userPrompt }]
    };
    const res = await fetch(cfg.endpoint, {
      method: "POST",
      signal,
      headers: {
        "content-type": "application/json",
        "x-api-key": cfg.apiKey,
        "anthropic-version": "2023-06-01",
        // Required when calling Anthropic from a browser-origin context.
        "anthropic-dangerous-direct-browser-access": "true"
      },
      body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error(`anthropic ${res.status}: ${await res.text()}`);
    const data = await res.json();
    const text = data.content?.find((b) => b.type === "text")?.text ?? "";
    return text;
  }
  async function callOpenAIResponses(cfg, userPrompt, signal) {
    const body = {
      model: cfg.model,
      input: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt }
      ],
      text: { format: { type: "json_object" } }
    };
    const res = await fetch(cfg.endpoint, {
      method: "POST",
      signal,
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${cfg.apiKey}`
      },
      body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error(`openai ${res.status}: ${await res.text()}`);
    const data = await res.json();
    if (typeof data.output_text === "string" && data.output_text) {
      return data.output_text;
    }
    for (const item of data.output ?? []) {
      for (const c of item.content ?? []) {
        if (c.type === "output_text" && c.text) return c.text;
      }
    }
    return "";
  }
  async function callChatCompletions(cfg, userPrompt, signal) {
    const body = {
      model: cfg.model,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt }
      ],
      temperature: 1,
      // Ensure JSON output (OpenAI standard, widely supported by compat servers)
      response_format: { type: "json_object" },
      // Disable reasoning/thinking for faster responses (Qwen-specific)
      chat_template_kwargs: { enable_thinking: false }
    };
    const res = await fetch(cfg.endpoint, {
      method: "POST",
      signal,
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${cfg.apiKey}`
      },
      body: JSON.stringify(body)
    });
    if (!res.ok) {
      throw new Error(`chat-completions ${res.status}: ${await res.text()}`);
    }
    const data = await res.json();
    return data.choices?.[0]?.message?.content ?? "";
  }
  function parseTranslations(raw, targetIndices) {
    const out = /* @__PURE__ */ new Map();
    if (!raw) {
      console.log("[drtv-en/bg] parseTranslations: empty raw");
      return out;
    }
    const json = extractJson(raw);
    let parsed;
    try {
      parsed = JSON.parse(json);
    } catch {
      console.log("[drtv-en/bg] parseTranslations: JSON parse failed, json=", json.slice(0, 200));
      return out;
    }
    if (!parsed || typeof parsed !== "object") return out;
    const obj = parsed;
    const translations = obj.translations && typeof obj.translations === "object" ? obj.translations : obj;
    const allow = new Set(targetIndices);
    for (const [key, value] of Object.entries(translations)) {
      const idx = Number(key);
      if (!Number.isInteger(idx) || !allow.has(idx)) continue;
      if (typeof value !== "string" || !value.trim()) continue;
      out.set(idx, value);
    }
    return out;
  }
  function extractJson(raw) {
    const start = raw.indexOf("{");
    const end = raw.lastIndexOf("}");
    if (start === -1 || end === -1 || end < start) return raw;
    return raw.slice(start, end + 1);
  }

  // src/background/cache.ts
  var DB_NAME = "drtv-en-translations";
  var STORE = "translations";
  var VERSION = 1;
  function makeKey(episodeId, hash, provider, model) {
    return `${episodeId}|${hash}|${provider}|${model}`;
  }
  function openDb() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          db.createObjectStore(STORE, { keyPath: "key" });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  async function getCachedCues(episodeId, hash, provider, model) {
    const db = await openDb();
    try {
      return await new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, "readonly");
        const req = tx.objectStore(STORE).get(makeKey(episodeId, hash, provider, model));
        req.onsuccess = () => {
          const entry = req.result;
          if (!entry) return resolve(null);
          try {
            resolve(JSON.parse(entry.cuesJson));
          } catch {
            resolve(null);
          }
        };
        req.onerror = () => reject(req.error);
      });
    } finally {
      db.close();
    }
  }
  async function putCachedCues(episodeId, hash, provider, model, cues) {
    const db = await openDb();
    try {
      await new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, "readwrite");
        const entry = {
          key: makeKey(episodeId, hash, provider, model),
          episodeId,
          sourceVttHash: hash,
          provider,
          model,
          cuesJson: JSON.stringify(cues),
          createdAt: Date.now()
        };
        tx.objectStore(STORE).put(entry);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } finally {
      db.close();
    }
  }
  async function sha256Hex(input) {
    const bytes = new TextEncoder().encode(input);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const arr = Array.from(new Uint8Array(digest));
    return arr.map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  // src/background/index.ts
  installVttSniffer();
  enableCorsProxy();
  chrome.action?.onClicked.addListener(() => {
    void chrome.runtime.openOptionsPage();
  });
  chrome.runtime.onMessage.addListener((msg, sender) => {
    if (!msg || typeof msg !== "object") return;
    const m = msg;
    const tabId = sender.tab?.id;
    if (m.type === "vtt-url" && m.url && tabId !== void 0) {
      setVttUrlForTab(tabId, m.url);
      console.log("[drtv-en/bg] captured VTT (page)", tabId, m.url);
    }
  });
  var activeJobs = /* @__PURE__ */ new Map();
  var heartbeatInterval = void 0;
  chrome.runtime.onConnect.addListener((port) => {
    if (port.name !== PORT_NAME) return;
    const tabId = port.sender?.tab?.id;
    if (tabId === void 0) {
      port.disconnect();
      return;
    }
    const send = (event) => {
      try {
        port.postMessage(event);
      } catch {
      }
    };
    port.onMessage.addListener((msg) => {
      if (msg.type === "episode-active") {
        cancelJob(tabId);
        return;
      }
      if (msg.type === "cancel") {
        cancelJob(tabId);
        return;
      }
      if (msg.type === "request-translate") {
        void runJob(tabId, msg.episodeId, msg.playhead ?? 0, send);
      }
      if (msg.type === "seek") {
        const job = activeJobs.get(tabId);
        if (job) job.playhead = msg.time;
      }
    });
    port.onDisconnect.addListener(() => {
      clearInterval(heartbeatInterval);
      cancelJob(tabId);
    });
  });
  function cancelJob(tabId) {
    const job = activeJobs.get(tabId);
    if (!job) return;
    job.abort.abort();
    activeJobs.delete(tabId);
  }
  async function runJob(tabId, episodeId, initialPlayhead, send) {
    cancelJob(tabId);
    const abort = new AbortController();
    const job = { episodeId, abort, playhead: initialPlayhead };
    activeJobs.set(tabId, job);
    heartbeatInterval = setInterval(() => {
      try {
        send({ type: "heartbeat" });
      } catch {
      }
    }, 1e4);
    try {
      const playlist = await waitForDanishPlaylist(tabId, abort.signal, send);
      if (!playlist) return;
      console.log("[drtv-en/bg] job", episodeId, "via playlist", playlist);
      send({ type: "status", state: "fetching-vtt", detail: playlist });
      const cues = await fetchCuesFromPlaylist(playlist, abort.signal);
      console.log("[drtv-en/bg] parsed cues:", cues.length, "first:", cues[0]);
      send({ type: "status", state: "parsing" });
      if (cues.length === 0) {
        send({ type: "error", message: "VTT parsed but no cues found" });
        return;
      }
      send({ type: "schedule", starts: cues.map((c) => c.start) });
      const cfg = await loadProviderConfig();
      if (!cfg.apiKey || !cfg.endpoint || !cfg.model) {
        send({
          type: "error",
          message: "Configure a provider and API key in the extension options first."
        });
        return;
      }
      const sourceHash = await sha256Hex(
        cues.map((c) => `${c.start}|${c.end}|${c.text}`).join("\n")
      );
      const cached = await getCachedCues(
        episodeId,
        sourceHash,
        cfg.provider,
        cfg.model
      ).catch((err) => {
        console.warn("[drtv-en/bg] cache lookup failed", err);
        return null;
      });
      if (cached) {
        console.log("[drtv-en/bg] cache hit:", cached.length, "cues");
        send({ type: "cues", cues: cached });
        send({ type: "done", total: cached.length });
        return;
      }
      send({ type: "status", state: "translating", detail: String(cues.length) });
      const translated = [];
      const onBatch = (batch) => {
        translated.push(...batch);
        send({ type: "cues", cues: batch });
      };
      const total = await translateWithLLM(cues, cfg, {
        onBatch,
        signal: abort.signal,
        getPlayhead: () => job.playhead
      });
      if (!abort.signal.aborted && translated.length > 0) {
        translated.sort((a, b) => a.start - b.start);
        void putCachedCues(
          episodeId,
          sourceHash,
          cfg.provider,
          cfg.model,
          translated
        ).catch((err) => console.warn("[drtv-en/bg] cache save failed", err));
      }
      send({ type: "done", total });
    } catch (err) {
      if (abort.signal.aborted) return;
      send({
        type: "error",
        message: err instanceof Error ? err.message : String(err)
      });
    } finally {
      clearInterval(heartbeatInterval);
      if (activeJobs.get(tabId)?.abort === abort) {
        activeJobs.delete(tabId);
      }
    }
  }
  async function waitForDanishPlaylist(tabId, signal, send) {
    send({ type: "status", state: "waiting-for-vtt" });
    const deadline = Date.now() + 1e4;
    while (!signal.aborted && Date.now() < deadline) {
      const subs = getSubsForTab(tabId);
      const masterUrl = subs ? deriveMasterUrl(subs) : void 0;
      if (masterUrl) {
        const resolved = await tryResolveFromMaster(masterUrl, signal);
        if (resolved) return resolved;
        const heuristic = subs ? chooseBestPlaylist(subs) : void 0;
        if (heuristic) {
          console.warn(
            "[drtv-en/bg] master parse failed, using heuristic playlist",
            heuristic
          );
          return heuristic;
        }
      }
      await new Promise((r) => setTimeout(r, 250));
    }
    if (signal.aborted) return void 0;
    send({
      type: "error",
      message: "Could not find a Danish subtitle track for this episode (no DRTV URLs sniffed)."
    });
    return void 0;
  }
  async function tryResolveFromMaster(masterUrl, signal) {
    try {
      const res = await fetch(masterUrl, { signal });
      if (!res.ok) {
        console.warn("[drtv-en/bg] master fetch", masterUrl, res.status);
        return void 0;
      }
      const tracks = parseSubtitleTracks(await res.text());
      const danish = pickDanishTrack(tracks);
      if (!danish || !danish.uri) {
        console.warn("[drtv-en/bg] no Danish track in master", tracks);
        return void 0;
      }
      const playlist = new URL(danish.uri, masterUrl).toString();
      console.log("[drtv-en/bg] resolved Danish track via master:", playlist);
      return playlist;
    } catch (err) {
      if (signal.aborted) return void 0;
      console.warn("[drtv-en/bg] master resolution failed", err);
      return void 0;
    }
  }
  chrome.webNavigation?.onHistoryStateUpdated.addListener(
    (details) => {
      if (details.frameId !== 0) return;
      clearVttForTab(details.tabId);
      cancelJob(details.tabId);
    },
    { url: [{ hostEquals: "www.dr.dk" }] }
  );
})();
//# sourceMappingURL=index.js.map
