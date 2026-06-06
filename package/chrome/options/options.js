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
async function saveProviderConfig(cfg) {
  await chrome.storage.local.set({ [KEY]: cfg });
}

// src/background/cache.ts
var DB_NAME = "drtv-en-translations";
var STORE = "translations";
var VERSION = 1;
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
async function getCacheStats() {
  const db = await openDb();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).getAll();
      req.onsuccess = () => {
        const all = req.result ?? [];
        let bytes = 0;
        for (const e of all) bytes += e.cuesJson.length;
        resolve({ entries: all.length, bytes });
      };
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}
async function clearCache() {
  const db = await openDb();
  try {
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}

// src/options/options.ts
function $(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
}
function $sel(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
}
function populateProviders() {
  const sel = $sel("provider");
  for (const [key, preset] of Object.entries(PROVIDER_PRESETS)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = preset.label;
    sel.appendChild(opt);
  }
}
var currentProvider = "anthropic";
function applyPresetIfDefault(next) {
  const prev = PROVIDER_PRESETS[currentProvider];
  const preset = PROVIDER_PRESETS[next];
  const endpointEl = $("endpoint");
  const modelEl = $("model");
  if (!endpointEl.value || endpointEl.value === prev.endpoint) {
    endpointEl.value = preset.endpoint;
  }
  if (!modelEl.value || modelEl.value === prev.model) {
    modelEl.value = preset.model;
  }
  currentProvider = next;
}
async function init() {
  populateProviders();
  const cfg = await loadProviderConfig();
  currentProvider = cfg.provider;
  $sel("provider").value = cfg.provider;
  $("endpoint").value = cfg.endpoint || PROVIDER_PRESETS[cfg.provider].endpoint;
  $("model").value = cfg.model || PROVIDER_PRESETS[cfg.provider].model;
  $("apiKey").value = cfg.apiKey;
  $sel("provider").addEventListener("change", () => {
    applyPresetIfDefault($sel("provider").value);
  });
  document.getElementById("save").addEventListener("click", async () => {
    const status = document.getElementById("status");
    status.classList.remove("error");
    const provider = $sel("provider").value;
    const prev = await loadProviderConfig();
    const next = {
      provider,
      endpoint: $("endpoint").value.trim(),
      model: $("model").value.trim(),
      apiKey: $("apiKey").value,
      batchSize: prev.batchSize,
      contextWindow: prev.contextWindow,
      maxParallel: prev.maxParallel
    };
    if (!next.endpoint || !next.model || !next.apiKey) {
      status.classList.add("error");
      status.textContent = "Endpoint, model, and API key are all required.";
      return;
    }
    await saveProviderConfig(next);
    status.textContent = "Saved.";
    setTimeout(() => status.textContent = "", 2e3);
  });
}
async function refreshCacheStats() {
  const el = document.getElementById("cacheStats");
  try {
    const { entries, bytes } = await getCacheStats();
    if (entries === 0) {
      el.textContent = "Nothing cached yet.";
    } else {
      el.textContent = `${entries} episode${entries === 1 ? "" : "s"} cached (${formatBytes(bytes)}).`;
    }
  } catch (err) {
    el.textContent = `Cache unavailable: ${err instanceof Error ? err.message : String(err)}`;
  }
}
function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
document.getElementById("clearCache")?.addEventListener("click", async () => {
  await clearCache();
  await refreshCacheStats();
});
void init().then(refreshCacheStats);
//# sourceMappingURL=options.js.map
