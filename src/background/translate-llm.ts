// Real LLM translator. Ports the prompt + batching strategy from
// src/drtv_in_english/llm.py, but the API call goes through one of
// three thin adapters depending on provider config:
//
//   - Anthropic       → POST /v1/messages
//   - OpenAI          → POST /v1/responses
//   - Gemini + compat → POST /v1/chat/completions
//
// Each batch translates `batchSize` cues with `contextWindow` cues of
// surround for ASR/idiom context. A pool of up to `maxParallel`
// batches run concurrently; each is yielded back through `onBatch` in
// completion order so the content script can stream cues. A failed
// batch falls back to original text (same as the Python backend),
// so one bad batch never kills the whole run.

import type { Cue } from "../shared/types.js";
import type { ProviderConfig } from "../shared/storage.js";
import { loadMaxParallel, saveMaxParallel } from "../shared/storage.js";

export interface TranslateLLMOptions {
  onBatch: (cues: Cue[]) => void;
  signal?: AbortSignal;
  // Returns the user's current playback position so we can translate
  // the cues they're about to hit first. Called fresh for each batch
  // pick, so a seek immediately reorders the queue.
  getPlayhead?: () => number;
}

const SYSTEM_PROMPT =
  "You are an expert translator and editor of Danish TV subtitles. " +
  "Your task is to: " +
  "1) Translate the corrected text into the requested target language. " +
  "2) Preserve the original meaning, speaker intent, and subtitle length. " +
  "3) Do not translate proper nouns (keep them as-is). " +
  "4) Return ONLY the requested JSON object, with no extra prose.";

// Upper bound on a single provider call. Without this a silent
// connection death leaves the user staring at the buffering overlay
// for many minutes — the fetch never settles. 90s comfortably covers
// slow legitimate responses; anything longer is almost certainly hung.
const PROVIDER_TIMEOUT_MS = 90_000;

function withTimeout(
  signal: AbortSignal | undefined,
  ms: number,
): { signal: AbortSignal; cancel: () => void } {
  const ctrl = new AbortController();
  const onAbort = () => ctrl.abort(signal!.reason);
  if (signal) {
    if (signal.aborted) ctrl.abort(signal.reason);
    else signal.addEventListener("abort", onAbort, { once: true });
  }
  const timer = setTimeout(
    () => ctrl.abort(new Error(`provider request timed out after ${ms}ms`)),
    ms,
  );
  return {
    signal: ctrl.signal,
    cancel: () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    },
  };
}

export async function translateWithLLM(
  cues: Cue[],
  cfg: ProviderConfig,
  opts: TranslateLLMOptions,
): Promise<number> {
  if (cues.length === 0) return 0;

  const batches = buildBatches(cues, cfg.batchSize, cfg.contextWindow);
  let emitted = 0;

  // Load persisted maxParallel for this provider+model, or use configured value
  const persistedMaxParallel = await loadMaxParallel(cfg.provider, cfg.model);
  let currentConcurrency = persistedMaxParallel ?? cfg.maxParallel;

  // Adaptive concurrency: track success rate and adjust
  const attemptWithConcurrency = async (
    concurrency: number,
  ): Promise<{ success: boolean; saved: boolean }> => {
    const remaining = new Set<number>(batches.map((_, i) => i));
    let roundFailures = 0;
    let roundTotal = 0;

    const pickNext = (): number | undefined => {
      if (remaining.size === 0) return undefined;
      const p = opts.getPlayhead?.() ?? 0;
      let bestI: number | undefined;
      let bestD = Infinity;
      for (const i of remaining) {
        const indices = batches[i]!.targetIndices;
        const firstCue = cues[indices[0]!]!;
        const lastCue = cues[indices[indices.length - 1]!]!;
        let d: number;
        if (firstCue.start <= p && p <= lastCue.end) {
          d = 0;
        } else if (firstCue.start >= p) {
          d = firstCue.start - p;
        } else {
          d = (p - firstCue.start) * 10 + 1000;
        }
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      if (bestI === undefined) return undefined;
      remaining.delete(bestI);
      return bestI;
    };

    const workers: Promise<void>[] = [];
    for (let w = 0; w < Math.min(concurrency, remaining.size); w++) {
      workers.push(
        (async () => {
          while (remaining.size > 0) {
            if (opts.signal?.aborted) return;
            const i = pickNext();
            if (i === undefined) return;
            const { targetIndices, userPrompt } = batches[i]!;
            roundTotal++;
            const translations = await runBatch(
              cfg,
              userPrompt,
              targetIndices,
              opts.signal,
            ).catch((err) => {
              console.warn(
                "[drtv-en/bg] batch failed",
                targetIndices,
                err,
              );
              roundFailures++;
              return new Map<number, string>();
            });
            if (opts.signal?.aborted) return;
            const batchCues: Cue[] = targetIndices.map((idx) => {
              const src = cues[idx]!;
              const text = translations.get(idx)?.trim() || src.text;
              return { start: src.start, end: src.end, text };
            });
            emitted += batchCues.length;
            opts.onBatch(batchCues);
          }
        })(),
      );
    }

    await Promise.all(workers);

    // If >50% failures, this concurrency level is too high
    const failureRate = roundTotal > 0 ? roundFailures / roundTotal : 0;
    const success = failureRate <= 0.5;

    // Save on first successful round
    let saved = false;
    if (success && persistedMaxParallel === null) {
      await saveMaxParallel(cfg.provider, cfg.model, concurrency);
      saved = true;
    }

    return { success, saved };
  };

  // Start with current concurrency, halve if needed
  while (currentConcurrency >= 1) {
    if (opts.signal?.aborted) break;
    const result = await attemptWithConcurrency(currentConcurrency);
    if (result.success) {
      return emitted;
    }
    // Halve concurrency and retry remaining batches
    currentConcurrency = Math.max(1, Math.floor(currentConcurrency / 2));
    console.log(
      "[drtv-en/bg] reducing concurrency to",
      currentConcurrency,
      "due to failures",
    );
  }

  return emitted;
}

interface Batch {
  targetIndices: number[];
  userPrompt: string;
}

function buildBatches(
  cues: Cue[],
  batchSize: number,
  contextWindow: number,
): Batch[] {
  const n = cues.length;
  const out: Batch[] = [];
  for (let start = 0; start < n; start += batchSize) {
    const end = Math.min(n, start + batchSize);
    const windowStart = Math.max(0, start - contextWindow);
    const windowEnd = Math.min(n, end + contextWindow);

    let numbered = "";
    for (let j = windowStart; j < windowEnd; j++) {
      const c = cues[j]!;
      const text = c.text || "[no text]";
      numbered += `${j}: [${c.start.toFixed(3)} -> ${c.end.toFixed(3)}] ${text}\n`;
    }

    const targetIndices: number[] = [];
    for (let i = start; i < end; i++) targetIndices.push(i);
    const targetIds = targetIndices.join(", ");

    const userPrompt =
      `Translate the Danish subtitles to English. Output ONLY English text.\n\n` +
      `Chunks (translate ONLY the chunks with ids ${targetIds}):\n\n` +
      `${numbered}` +
      `---\n` +
      `Return a JSON object of the form ` +
      `{"translations": {"<id>": "<corrected and translated text>", ...}} ` +
      `covering exactly the requested ids (${targetIds}).`;

    out.push({ targetIndices, userPrompt });
  }
  return out;
}

async function runBatch(
  cfg: ProviderConfig,
  userPrompt: string,
  targetIndices: number[],
  signal: AbortSignal | undefined,
): Promise<Map<number, string>> {
  const raw = await callProvider(cfg, userPrompt, signal);
  console.log("[drtv-en/bg] llm raw response (truncated):", raw?.slice(0, 500));
  const parsed = parseTranslations(raw, targetIndices);
  console.log("[drtv-en/bg] llm parsed:", parsed.size, "cues");
  return parsed;
}

async function callProvider(
  cfg: ProviderConfig,
  userPrompt: string,
  signal: AbortSignal | undefined,
): Promise<string> {
  const { signal: boundedSignal, cancel } = withTimeout(
    signal,
    PROVIDER_TIMEOUT_MS,
  );
  try {
    switch (cfg.provider) {
      case "anthropic":
        return await callAnthropic(cfg, userPrompt, boundedSignal);
      case "openai":
        return await callOpenAIResponses(cfg, userPrompt, boundedSignal);
      case "gemini":
      case "alx":
      case "openai-compatible":
        return await callChatCompletions(cfg, userPrompt, boundedSignal);
      default:
        throw new Error(`unsupported provider: ${cfg.provider}`);
    }
  } finally {
    cancel();
  }
}

async function callAnthropic(
  cfg: ProviderConfig,
  userPrompt: string,
  signal: AbortSignal | undefined,
): Promise<string> {
  const body = {
    model: cfg.model,
    max_tokens: 4096,
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: userPrompt }],
  };
  const res = await fetch(cfg.endpoint, {
    method: "POST",
    signal,
    headers: {
      "content-type": "application/json",
      "x-api-key": cfg.apiKey,
      "anthropic-version": "2023-06-01",
      // Required when calling Anthropic from a browser-origin context.
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`anthropic ${res.status}: ${await res.text()}`);
  const data = (await res.json()) as {
    content?: { type: string; text?: string }[];
  };
  const text = data.content?.find((b) => b.type === "text")?.text ?? "";
  return text;
}

async function callOpenAIResponses(
  cfg: ProviderConfig,
  userPrompt: string,
  signal: AbortSignal | undefined,
): Promise<string> {
  const body = {
    model: cfg.model,
    input: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ],
    text: { format: { type: "json_object" } },
  };
  const res = await fetch(cfg.endpoint, {
    method: "POST",
    signal,
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`openai ${res.status}: ${await res.text()}`);
  const data = (await res.json()) as {
    output_text?: string;
    output?: { content?: { type: string; text?: string }[] }[];
  };
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

async function callChatCompletions(
  cfg: ProviderConfig,
  userPrompt: string,
  signal: AbortSignal | undefined,
): Promise<string> {
  const body: Record<string, unknown> = {
    model: cfg.model,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ],
    temperature: 1.0,
    // Ensure JSON output (OpenAI standard, widely supported by compat servers)
    response_format: { type: "json_object" },
    // Disable reasoning/thinking for faster responses (Qwen-specific)
    chat_template_kwargs: { enable_thinking: false },
  };
  const res = await fetch(cfg.endpoint, {
    method: "POST",
    signal,
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`chat-completions ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as {
    choices?: { message?: { content?: string } }[];
  };
  return data.choices?.[0]?.message?.content ?? "";
}

function parseTranslations(
  raw: string,
  targetIndices: number[],
): Map<number, string> {
  const out = new Map<number, string>();
  if (!raw) {
    console.log("[drtv-en/bg] parseTranslations: empty raw");
    return out;
  }
  const json = extractJson(raw);
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    console.log("[drtv-en/bg] parseTranslations: JSON parse failed, json=", json.slice(0, 200));
    return out;
  }
  if (!parsed || typeof parsed !== "object") return out;
  const obj = parsed as Record<string, unknown>;
  const translations =
    obj.translations && typeof obj.translations === "object"
      ? (obj.translations as Record<string, unknown>)
      : obj;
  const allow = new Set(targetIndices);
  for (const [key, value] of Object.entries(translations)) {
    const idx = Number(key);
    if (!Number.isInteger(idx) || !allow.has(idx)) continue;
    if (typeof value !== "string" || !value.trim()) continue;
    out.set(idx, value);
  }
  return out;
}

function extractJson(raw: string): string {
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start === -1 || end === -1 || end < start) return raw;
  return raw.slice(start, end + 1);
}
