// Options page: pick a provider (which preloads default endpoint +
// model), edit any field, and persist to chrome.storage.local. When
// the user switches provider, fields that still hold the previous
// preset's defaults are updated to the new preset; fields the user
// has customized are left alone.

import {
  PROVIDER_PRESETS,
  loadProviderConfig,
  saveProviderConfig,
  type Provider,
  type ProviderConfig,
} from "../shared/storage.js";
import { clearCache, getCacheStats } from "../background/cache.js";

function $(id: string): HTMLInputElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as HTMLInputElement;
}

function $sel(id: string): HTMLSelectElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as HTMLSelectElement;
}

function populateProviders(): void {
  const sel = $sel("provider");
  for (const [key, preset] of Object.entries(PROVIDER_PRESETS)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = preset.label;
    sel.appendChild(opt);
  }
}

let currentProvider: Provider = "anthropic";

function applyPresetIfDefault(next: Provider): void {
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

async function init(): Promise<void> {
  populateProviders();
  const cfg = await loadProviderConfig();
  currentProvider = cfg.provider;

  $sel("provider").value = cfg.provider;
  $("endpoint").value = cfg.endpoint || PROVIDER_PRESETS[cfg.provider].endpoint;
  $("model").value = cfg.model || PROVIDER_PRESETS[cfg.provider].model;
  $("apiKey").value = cfg.apiKey;

  $sel("provider").addEventListener("change", () => {
    applyPresetIfDefault($sel("provider").value as Provider);
  });

  document.getElementById("save")!.addEventListener("click", async () => {
    const status = document.getElementById("status")!;
    status.classList.remove("error");
    const provider = $sel("provider").value as Provider;
    const prev = await loadProviderConfig();
    const next: ProviderConfig = {
      provider,
      endpoint: $("endpoint").value.trim(),
      model: $("model").value.trim(),
      apiKey: $("apiKey").value,
      batchSize: prev.batchSize,
      contextWindow: prev.contextWindow,
      maxParallel: prev.maxParallel,
    };
    if (!next.endpoint || !next.model || !next.apiKey) {
      status.classList.add("error");
      status.textContent =
        "Endpoint, model, and API key are all required.";
      return;
    }
    await saveProviderConfig(next);
    status.textContent = "Saved.";
    setTimeout(() => (status.textContent = ""), 2000);
  });
}

async function refreshCacheStats(): Promise<void> {
  const el = document.getElementById("cacheStats")!;
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

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

document.getElementById("clearCache")?.addEventListener("click", async () => {
  await clearCache();
  await refreshCacheStats();
});

void init().then(refreshCacheStats);
