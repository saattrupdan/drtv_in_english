// Provider configuration persisted in chrome.storage.local. Each
// preset binds an API shape (/messages, /responses, /chat/completions)
// to a default endpoint and model; both are user-editable.

export type Provider =
  | "anthropic"
  | "openai"
  | "gemini"
  | "openai-compatible";

export interface ProviderConfig {
  provider: Provider;
  endpoint: string;
  apiKey: string;
  model: string;
  batchSize: number;
  contextWindow: number;
  maxParallel: number;
}

export interface ProviderPreset {
  label: string;
  endpoint: string;
  model: string;
}

export const PROVIDER_PRESETS: Record<Provider, ProviderPreset> = {
  anthropic: {
    label: "Anthropic",
    endpoint: "https://api.anthropic.com/v1/messages",
    model: "claude-haiku-4-5",
  },
  openai: {
    label: "OpenAI",
    endpoint: "https://api.openai.com/v1/responses",
    model: "gpt-5-mini",
  },
  gemini: {
    label: "Gemini",
    endpoint:
      "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    model: "gemini-3.5-flash",
  },
  "openai-compatible": {
    label: "OpenAI-compatible",
    endpoint: "",
    model: "",
  },
};

const DEFAULTS: ProviderConfig = {
  provider: "anthropic",
  endpoint: PROVIDER_PRESETS.anthropic.endpoint,
  apiKey: "",
  model: PROVIDER_PRESETS.anthropic.model,
  batchSize: 5,
  contextWindow: 6,
  maxParallel: 20,
};

const KEY = "providerConfig";

export async function loadProviderConfig(): Promise<ProviderConfig> {
  const got = await chrome.storage.local.get(KEY);
  return { ...DEFAULTS, ...(got[KEY] ?? {}) };
}

export async function saveProviderConfig(cfg: ProviderConfig): Promise<void> {
  await chrome.storage.local.set({ [KEY]: cfg });
}
