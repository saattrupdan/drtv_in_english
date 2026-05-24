<script setup lang="ts">
import Hls from "hls.js";
import { computed, nextTick, onBeforeUnmount, ref, useTemplateRef, watch } from "vue";

type Stage = "idle" | "preparing" | "playing" | "error";

interface PrepareResponse {
  job_id: string;
  title: string;
  hls_url: string;
  original_subs_url: string;
  cue_count: number;
}

interface CueEvent {
  index: number;
  start: number;
  end: number;
  text: string;
  done: boolean;
}

const stage = ref<Stage>("idle");
const url = ref("");
const errorMessage = ref<string | null>(null);
const prepared = ref<PrepareResponse | null>(null);
const cuesReceived = ref(0);
const translationDone = ref(false);

const videoEl = useTemplateRef<HTMLVideoElement>("videoEl");

let hls: Hls | null = null;
let prepareAbort: AbortController | null = null;
let translateAbort: AbortController | null = null;
let englishTrack: TextTrack | null = null;

const canSubmit = computed(() => url.value.trim().length > 0);

const translationProgress = computed(() => {
  if (!prepared.value || prepared.value.cue_count === 0) return 0;
  return Math.min(
    100,
    Math.round((cuesReceived.value / prepared.value.cue_count) * 100),
  );
});

async function startPreparing() {
  if (!canSubmit.value) return;
  stage.value = "preparing";
  errorMessage.value = null;
  prepared.value = null;
  cuesReceived.value = 0;
  translationDone.value = false;

  prepareAbort = new AbortController();
  try {
    const response = await fetch("/api/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url.value.trim(), language: "en" }),
      signal: prepareAbort.signal,
    });
    if (!response.ok) {
      const detail = await safeDetail(response);
      throw new Error(detail ?? `Request failed (${response.status})`);
    }
    prepared.value = (await response.json()) as PrepareResponse;
    stage.value = "playing";
    await nextTick(); // Wait for <video> element to mount
    attachPlayer(videoEl.value!, prepared.value);
    startTranslationStream(prepared.value.job_id);
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    stage.value = "error";
    errorMessage.value = (err as Error).message || "Something went wrong.";
  }
}

async function safeDetail(response: Response): Promise<string | null> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body?.detail ?? null;
  } catch {
    return null;
  }
}

watch(
  () => [stage.value, prepared.value, videoEl.value] as const,
  ([currentStage, currentPrepared, currentVideo]) => {
    if (
      currentStage === "playing" &&
      currentPrepared !== null &&
      currentVideo !== null
    ) {
      attachPlayer(currentVideo, currentPrepared);
    }
  },
);

function attachPlayer(video: HTMLVideoElement, info: PrepareResponse) {
  teardownPlayer();

  if (Hls.isSupported()) {
    hls = new Hls();
    hls.loadSource(info.hls_url);
    hls.attachMedia(video);
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = info.hls_url;
  } else {
    stage.value = "error";
    errorMessage.value = "Your browser does not support HLS playback.";
    return;
  }

  englishTrack = video.addTextTrack("subtitles", "English", "en");
  englishTrack.mode = "hidden";

  video.addEventListener("loadedmetadata", () => {
    const danishTrack = Array.from(video.textTracks).find(
      (t) => t.language === "da",
    );
    if (danishTrack) danishTrack.mode = "showing";
  });

  void video.play().catch(() => {
    // Autoplay blocked — user can press play. No-op.
  });
}

function startTranslationStream(jobId: string) {
  translateAbort = new AbortController();
  void consumeTranslations(jobId, translateAbort.signal);
}

async function consumeTranslations(jobId: string, signal: AbortSignal) {
  try {
    const response = await fetch(`/api/translate/${jobId}`, { signal });
    if (!response.ok || !response.body) {
      throw new Error(`Translation stream failed (${response.status})`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line.length === 0) continue;
        handleCueEvent(JSON.parse(line) as CueEvent);
      }
    }
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    // Soft-fail: video keeps playing with Danish subs.
    console.error("Translation stream error:", err);
  }
}

function handleCueEvent(event: CueEvent) {
  if (event.done) {
    translationDone.value = true;
    return;
  }
  if (!englishTrack || !videoEl.value) {
    console.warn("Cue received before player attached, dropping:", event);
    return;
  }

  const cue = new VTTCue(event.start, event.end, event.text);
  englishTrack.addCue(cue);
  cuesReceived.value += 1;

  if (!englishTrack.mode || englishTrack.mode === "hidden") {
    englishTrack.mode = "showing";
    const danishTrack = Array.from(videoEl.value.textTracks).find(
      (t) => t.language === "da",
    );
    if (danishTrack) danishTrack.mode = "hidden";
  }
}

function teardownPlayer() {
  if (hls) {
    hls.destroy();
    hls = null;
  }
  englishTrack = null;
}

function onSubmit() {
  void startPreparing();
}

function goHome() {
  if (prepareAbort) {
    prepareAbort.abort();
    prepareAbort = null;
  }
  if (translateAbort) {
    translateAbort.abort();
    translateAbort = null;
  }
  teardownPlayer();
  stage.value = "idle";
  url.value = "";
  errorMessage.value = null;
  prepared.value = null;
  cuesReceived.value = 0;
  translationDone.value = false;
}

onBeforeUnmount(() => {
  if (prepareAbort) prepareAbort.abort();
  if (translateAbort) translateAbort.abort();
  teardownPlayer();
});
</script>

<template>
  <div class="page">
    <header class="header" :class="{ 'header--compact': stage !== 'idle' }">
      <svg
        class="logo"
        role="img"
        aria-labelledby="DRTV-logo"
        viewBox="0 0 800 800"
      >
        <title id="DRTV-logo">DRTV</title>
        <path d="M0 0v800h800V0Z" fill="#ff001e"></path>
        <g>
          <path d="M0 560h800v240H0z"></path>
          <path
            fill="#fff"
            d="M319.27 618.67h-171.2a2.07 2.07 0 0 0-2.28 2.27v115.3a2.09 2.09 0 0 0 2.28 2.29h171.2c50.29 0 75.67-16.71 75.67-60.31 0-43.3-25.38-59.55-75.67-59.55Zm-44.33 97.83h-53.63c-1.52 0-1.83-.61-1.83-1.83v-72.16c0-1.2.31-1.82 1.83-1.82h53.63c31.89 0 44.66 9.11 44.66 37.83s-12.77 37.98-44.66 37.98ZM665 733.67l-49.52-34.94c-1.22-.92-2-1.37-2-2s.46-.92 1.53-.92c25 0 44.5-11.69 44.5-37.51s-16.65-39.63-47.51-39.63H424.4a2.06 2.06 0 0 0-2.27 2.27v115.3a2.08 2.08 0 0 0 2.27 2.29h69a2.08 2.08 0 0 0 2.28-2.29V708.9c0-1.37.46-1.82 1.83-1.82h39.73c2 0 2.73.15 4.1 1.22l34.49 28.56a6.6 6.6 0 0 0 4.85 1.67h85.84q1.83 0 1.83-1.38c.04-1.21-2.09-2.57-3.35-3.48Zm-107.54-48h-59.81c-1.37 0-1.83-.46-1.83-1.82v-41.34c0-1.36.46-1.82 1.83-1.82h59.85c20.06 0 28.56 5.48 28.56 22 0 16.74-8.5 22.97-28.56 22.97Z"
          ></path>
        </g>
        <g>
          <path
            fill="#fff"
            d="M352.12 349.76 474 279.38a4.92 4.92 0 0 0 0-8.52l-121.9-70.37a4.91 4.91 0 0 0-7.37 4.26V345.5a4.92 4.92 0 0 0 7.39 4.26Z"
          ></path>
          <path
            fill="#fff"
            d="M396.59 467.42a191.8 191.8 0 0 1-176.88-117.14l44.82-19a143.2 143.2 0 0 0 132.06 87.45c79.06 0 143.37-64.31 143.37-143.36S475.65 132 396.59 132a143.2 143.2 0 0 0-132.11 87.57l-44.83-19A191.77 191.77 0 0 1 396.59 83.32c105.9 0 192.05 86.15 192.05 192.05s-86.15 192.05-192.05 192.05Z"
          ></path>
        </g>
      </svg>
      <h1 class="title">DRTV <span class="title-accent">in English</span></h1>
    </header>

    <main class="main">
      <section v-if="stage === 'idle'" class="card">
        <p class="lede">
          Paste a DRTV URL and watch it with English subtitles.
        </p>

        <form class="form" @submit.prevent="onSubmit">
          <input
            v-model="url"
            type="text"
            class="url-input"
            placeholder="https://www.dr.dk/drtv/se/…"
            autofocus
            @keydown.enter.prevent="onSubmit"
          />

          <button type="submit" class="primary-button" :disabled="!canSubmit">
            Watch with English subs
          </button>
        </form>
      </section>

      <section v-else-if="stage === 'preparing'" class="card card--center">
        <div class="preparing">
          <div class="spinner" aria-hidden="true"></div>
          <span class="preparing-label">Preparing stream…</span>
        </div>
      </section>

      <section
        v-else-if="stage === 'playing' && prepared"
        class="card card--player"
      >
        <video ref="videoEl" class="video" controls playsinline>
          <track
            kind="subtitles"
            srclang="da"
            :src="prepared.original_subs_url"
            label="Dansk"
            default
          />
        </video>

        <div v-if="!translationDone" class="translation-status">
          <div class="spinner spinner--sm" aria-hidden="true"></div>
          <span>
            Translating subtitles… {{ cuesReceived }} /
            {{ prepared.cue_count }} ({{ translationProgress }}%)
          </span>
        </div>
        <div v-else class="translation-status translation-status--done">
          English subtitles ready.
        </div>

        <button class="ghost-button home-button" @click="goHome">
          ← Watch another video
        </button>
      </section>

      <section v-else-if="stage === 'error'" class="card card--center">
        <div class="error-block">
          <p class="error-message">{{ errorMessage }}</p>
          <button class="ghost-button" @click="goHome">← Try again</button>
        </div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.page {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 48px 24px 80px;
}

.header {
  display: flex;
  align-items: center;
  gap: 16px;
  transition:
    transform 600ms cubic-bezier(0.22, 1, 0.36, 1),
    margin 600ms cubic-bezier(0.22, 1, 0.36, 1);
  margin-bottom: 48px;
}

.header--compact {
  margin-bottom: 24px;
}

.logo {
  width: 64px;
  height: 64px;
  border-radius: 4px;
}

.title {
  font-size: 28px;
  font-weight: 800;
  margin: 0;
  letter-spacing: -0.04em;
  text-transform: uppercase;
}

.title-accent {
  color: var(--accent);
}

.main {
  width: 100%;
  max-width: 720px;
}

.card {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 32px;
  box-shadow: var(--shadow-lg);
}

.card--center {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
}

.card--player {
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.lede {
  margin: 0 0 24px;
  color: var(--text-muted);
  font-size: 15px;
  line-height: 1.5;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.url-input {
  width: 100%;
  background: var(--bg-elev-2);
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  outline: none;
  color: var(--text);
  font-size: 16px;
  padding: 14px 16px;
  transition: border-color 200ms ease;
}

.url-input:focus {
  border-color: var(--accent);
}

.url-input::placeholder {
  color: var(--text-muted);
}

.primary-button {
  background: var(--accent);
  color: white;
  border: none;
  padding: 14px 20px;
  border-radius: var(--radius);
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition:
    background 150ms ease,
    transform 150ms ease;
}

.primary-button:hover:not(:disabled) {
  background: var(--accent-hover);
}

.primary-button:active:not(:disabled) {
  transform: scale(0.99);
}

.primary-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.ghost-button {
  background: transparent;
  color: var(--text-muted);
  border: 1px solid var(--border);
  padding: 8px 14px;
  border-radius: 8px;
  font-size: 13px;
  cursor: pointer;
  align-self: flex-start;
  transition:
    color 150ms ease,
    border-color 150ms ease;
}

.ghost-button:hover {
  color: var(--text);
  border-color: var(--text-muted);
}

.preparing {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--text-muted);
}

.preparing-label {
  font-size: 14px;
}

.spinner {
  width: 22px;
  height: 22px;
  border: 2.5px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 800ms linear infinite;
}

.spinner--sm {
  width: 14px;
  height: 14px;
  border-width: 2px;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.video {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  border-radius: var(--radius);
  display: block;
}

.translation-status {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}

.translation-status--done {
  color: var(--success, #22c55e);
}

.home-button {
  align-self: flex-start;
}

.error-block {
  display: flex;
  flex-direction: column;
  gap: 16px;
  align-items: center;
  text-align: center;
}

.error-message {
  color: #ef4444;
  margin: 0;
  font-size: 14px;
}
</style>
