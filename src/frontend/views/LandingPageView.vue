<script setup lang="ts">
import { computed, onBeforeUnmount, ref, useTemplateRef } from "vue";

type Stage = "idle" | "processing" | "ready" | "playing" | "error";

interface VideoWithSubs {
  video_path: string;
  subtitles_path: string;
}

interface ProgressEvent {
  stage: string;
  percentage: number;
  message?: string | null;
  result?: VideoWithSubs | null;
}

const stage = ref<Stage>("idle");
const url = ref("");
const progress = ref(0);
const progressMessage = ref<string | null>(null);
const errorMessage = ref<string | null>(null);
const showToast = ref(false);
const videoSrc = ref<string | null>(null);
const subtitlesSrc = ref<string | null>(null);

const videoEl = useTemplateRef<HTMLVideoElement>("videoEl");

let toastTimer: number | null = null;
let abortController: AbortController | null = null;

const canSubmit = computed(() => url.value.trim().length > 0);

const progressLabel = computed(() => {
  if (progressMessage.value) return progressMessage.value;
  if (progress.value < 50) return "Downloading video…";
  if (progress.value < 100) return "Translating subtitles…";
  return "Done";
});

async function startProcessing() {
  if (!canSubmit.value) return;
  stage.value = "processing";
  progress.value = 0;
  progressMessage.value = null;
  errorMessage.value = null;
  videoSrc.value = null;
  subtitlesSrc.value = null;

  abortController = new AbortController();
  try {
    const response = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url.value.trim(), language: "en" }),
      signal: abortController.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error(`Request failed (${response.status})`);
    }
    await consumeStream(response.body);
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    stage.value = "error";
    errorMessage.value = (err as Error).message || "Something went wrong.";
  }
}

async function consumeStream(body: ReadableStream<Uint8Array>) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newlineIndex;
    while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line.length === 0) continue;
      handleEvent(JSON.parse(line) as ProgressEvent);
    }
  }
  const tail = buffer.trim();
  if (tail.length > 0) handleEvent(JSON.parse(tail) as ProgressEvent);
}

function handleEvent(event: ProgressEvent) {
  if (event.stage === "error") {
    stage.value = "error";
    errorMessage.value = event.message || "Pipeline failed.";
    return;
  }
  progress.value = Math.max(progress.value, event.percentage);
  progressMessage.value = event.message ?? null;
  if (event.stage === "completed" && event.result) {
    videoSrc.value = event.result.video_path;
    subtitlesSrc.value = event.result.subtitles_path;
    onComplete();
  }
}

function onComplete() {
  stage.value = "ready";
  showToast.value = true;
  toastTimer = window.setTimeout(() => {
    showToast.value = false;
  }, 4000);
}

function onSubmit() {
  void startProcessing();
}

function goHome() {
  if (abortController !== null) {
    abortController.abort();
    abortController = null;
  }
  if (toastTimer !== null) {
    window.clearTimeout(toastTimer);
    toastTimer = null;
  }
  stage.value = "idle";
  url.value = "";
  progress.value = 0;
  progressMessage.value = null;
  errorMessage.value = null;
  videoSrc.value = null;
  subtitlesSrc.value = null;
  showToast.value = false;
}

function onVideoPlay() {
  stage.value = "playing";
  showToast.value = false;
  const el = videoEl.value;
  if (el && el.requestFullscreen) {
    el.requestFullscreen().catch(() => {
      // ignore — some browsers block this without a real user gesture
    });
  }
}

function onFullscreenChange() {
  if (!document.fullscreenElement && stage.value === "playing") {
    stage.value = "playing";
  }
}

document.addEventListener("fullscreenchange", onFullscreenChange);

onBeforeUnmount(() => {
  if (abortController !== null) abortController.abort();
  if (toastTimer !== null) window.clearTimeout(toastTimer);
  document.removeEventListener("fullscreenchange", onFullscreenChange);
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
      <!-- IDLE -->
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

      <!-- PROCESSING -->
      <section v-else-if="stage === 'processing'" class="card card--center">
        <div class="progress-wrapper progress-wrapper--center">
          <div class="progress-meta">
            <span class="progress-label">{{ progressLabel }}</span>
            <span class="progress-percent">{{ Math.floor(progress) }}%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" :style="{ width: `${progress}%` }"></div>
          </div>
        </div>
      </section>

      <!-- READY / PLAYING -->
      <section
        v-else-if="stage === 'ready' || stage === 'playing'"
        class="card card--player"
        :class="{ 'card--playing': stage === 'playing' }"
      >
        <div
          v-if="stage === 'ready'"
          class="progress-wrapper progress-wrapper--top"
        >
          <div class="progress-meta">
            <span class="progress-label">Done</span>
            <span class="progress-percent">100%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" style="width: 100%"></div>
          </div>
        </div>

        <video
          ref="videoEl"
          class="video"
          controls
          playsinline
          :src="videoSrc ?? undefined"
          @play="onVideoPlay"
        >
          <track
            v-if="subtitlesSrc"
            kind="subtitles"
            srclang="en"
            :src="subtitlesSrc"
            label="English"
            default
          />
        </video>

        <button class="ghost-button home-button" @click="goHome">
          ← Watch another video
        </button>
      </section>

      <!-- ERROR -->
      <section v-else-if="stage === 'error'" class="card card--center">
        <div class="error-block">
          <p class="error-message">{{ errorMessage }}</p>
          <button class="ghost-button" @click="goHome">← Try again</button>
        </div>
      </section>
    </main>

    <transition name="toast">
      <div v-if="showToast" class="toast" role="status">
        <span class="toast-dot"></span>
        Ready to watch!
      </div>
    </transition>
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

.card--playing {
  background: transparent;
  border: none;
  box-shadow: none;
  padding: 0;
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

.progress-wrapper {
  width: 100%;
  transition: all 600ms cubic-bezier(0.22, 1, 0.36, 1);
}

.progress-wrapper--center {
  max-width: 480px;
}

.progress-meta {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
  font-size: 13px;
}

.progress-label {
  color: var(--text-muted);
}

.progress-percent {
  color: var(--text);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.progress-track {
  height: 6px;
  background: var(--bg-elev-2);
  border-radius: 999px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 999px;
  transition: width 120ms linear;
}

.video {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  border-radius: var(--radius);
  display: block;
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

.toast {
  position: fixed;
  bottom: 32px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg-elev-2);
  color: var(--text);
  padding: 12px 18px;
  border-radius: 999px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow-lg);
  font-size: 14px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 10px;
}

.toast-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.2);
}

.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translate(-50%, 12px);
}

.toast-enter-active,
.toast-leave-active {
  transition:
    opacity 300ms ease,
    transform 300ms ease;
}
</style>
