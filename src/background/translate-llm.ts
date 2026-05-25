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
  "The source text was transcribed by an ASR model and may contain errors " +
  "(missing words, mangled proper nouns, literal translations of idioms). " +
  "Your task is to: " +
  "1) Correct any ASR errors using the surrounding context. " +
  "2) Translate the corrected text into the requested target language. " +
  "3) Preserve the original meaning, speaker intent, and subtitle length. " +
  "4) Do not translate proper nouns (keep them as-is). " +
  "5) Return ONLY the requested JSON object, with no extra prose.";

const TARGET_LANGUAGE = "en";

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

  // Remaining batch indices, re-prioritized on each pick by distance
  // from the user's current playhead. A seek immediately surfaces the
  // batches around the new position; finished batches stay finished.
  const remaining = new Set<number>(batches.map((_, i) => i));

  const pickNext = (): number | undefined => {
    if (remaining.size === 0) return undefined;
    const p = opts.getPlayhead?.() ?? 0;
    let bestI: number | undefined;
    let bestD = Infinity;
    for (const i of remaining) {
      const indices = batches[i]!.targetIndices;
      const firstCue = cues[indices[0]!]!;
      const lastCue = cues[indices[indices.length - 1]!]!;
      // Batches that straddle the playhead translate the cue the user
      // needs *right now* — always pick those first. Otherwise: future
      // batches preferred; past batches still translated but
      // deprioritized (the user may scroll back to them later).
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

  const concurrency = Math.max(1, cfg.maxParallel);
  const workers: Promise<void>[] = [];

  for (let w = 0; w < Math.min(concurrency, batches.length); w++) {
    workers.push(
      (async () => {
        while (remaining.size > 0) {
          if (opts.signal?.aborted) return;
          const i = pickNext();
          if (i === undefined) return;
          const { targetIndices, userPrompt } = batches[i]!;
          const translations = await runBatch(
            cfg,
            userPrompt,
            targetIndices,
            opts.signal,
          ).catch((err) => {
            console.warn("[drtv-en/bg] batch failed", targetIndices, err);
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
      `Target language: ${TARGET_LANGUAGE}\n\n` +
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
  return parseTranslations(raw, targetIndices);
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
  const body = {
    model: cfg.model,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ],
    response_format: { type: "json_object" },
    temperature: 1.0,
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
  if (!raw) return out;
  const json = extractJson(raw);
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
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
