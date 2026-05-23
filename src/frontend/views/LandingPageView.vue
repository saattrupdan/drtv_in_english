<script setup lang="ts">
import { computed, onBeforeUnmount, ref, useTemplateRef, watch } from "vue";

type Stage = "idle" | "processing" | "ready" | "playing";

const stage = ref<Stage>("idle");
const url = ref("");
const progress = ref(0);
const showToast = ref(false);
const isDragOver = ref(false);
const selectedFileName = ref<string | null>(null);

const fileInput = useTemplateRef<HTMLInputElement>("fileInput");
const videoEl = useTemplateRef<HTMLVideoElement>("videoEl");

let progressTimer: number | null = null;
let toastTimer: number | null = null;

const canSubmit = computed(
  () => url.value.trim().length > 0 || selectedFileName.value !== null,
);

const progressLabel = computed(() => {
  if (progress.value < 50) return "Downloading video…";
  if (progress.value < 100) return "Transcribing audio…";
  return "Done";
});

function startProcessing() {
  if (!canSubmit.value) return;
  stage.value = "processing";
  progress.value = 0;
  progressTimer = window.setInterval(() => {
    const step = progress.value < 50 ? 1.2 : 0.9;
    progress.value = Math.min(100, progress.value + step);
    if (progress.value >= 100) {
      if (progressTimer !== null) {
        window.clearInterval(progressTimer);
        progressTimer = null;
      }
      onComplete();
    }
  }, 80);
}

function onComplete() {
  stage.value = "ready";
  showToast.value = true;
  toastTimer = window.setTimeout(() => {
    showToast.value = false;
  }, 4000);
}

function onSubmit() {
  startProcessing();
}

function onFilePicked(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (file) {
    selectedFileName.value = file.name;
    url.value = "";
  }
}

function onDrop(event: DragEvent) {
  isDragOver.value = false;
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    selectedFileName.value = file.name;
    url.value = "";
  }
}

function onDragOver() {
  isDragOver.value = true;
}

function onDragLeave() {
  isDragOver.value = false;
}

function openFilePicker() {
  fileInput.value?.click();
}

function clearFile() {
  selectedFileName.value = null;
  if (fileInput.value) fileInput.value.value = "";
}

function goHome() {
  if (progressTimer !== null) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
  if (toastTimer !== null) {
    window.clearTimeout(toastTimer);
    toastTimer = null;
  }
  stage.value = "idle";
  url.value = "";
  progress.value = 0;
  showToast.value = false;
  selectedFileName.value = null;
  if (fileInput.value) fileInput.value.value = "";
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
    // User exited fullscreen — keep playing inline (per spec)
    stage.value = "playing";
  }
}

document.addEventListener("fullscreenchange", onFullscreenChange);

watch(url, (val) => {
  if (val.length > 0 && selectedFileName.value) {
    selectedFileName.value = null;
    if (fileInput.value) fileInput.value.value = "";
  }
});

onBeforeUnmount(() => {
  if (progressTimer !== null) window.clearInterval(progressTimer);
  if (toastTimer !== null) window.clearTimeout(toastTimer);
  document.removeEventListener("fullscreenchange", onFullscreenChange);
});
</script>

<template>
  <div class="page">
    <header class="header" :class="{ 'header--compact': stage !== 'idle' }">
      <img
        src="/but-with-subs-logo.jpg"
        alt="But With Subs logo"
        class="logo"
      />
      <h1 class="title">
        <span class="title-ellipsis">...</span> But With
        <span class="title-accent">Subs</span>
      </h1>
    </header>

    <main class="main">
      <!-- IDLE -->
      <section v-if="stage === 'idle'" class="card">
        <p class="lede">
          Watch anything with subtitles in the language you actually want.
        </p>

        <form class="form" @submit.prevent="onSubmit">
          <div
            class="dropzone"
            :class="{ 'dropzone--over': isDragOver }"
            @dragover.prevent="onDragOver"
            @dragleave.prevent="onDragLeave"
            @drop.prevent="onDrop"
          >
            <input
              v-model="url"
              type="text"
              class="url-input"
              placeholder="Paste a video URL…"
              autofocus
              @keydown.enter.prevent="onSubmit"
            />

            <div class="dropzone-actions">
              <span class="dropzone-hint">
                or drop a file here ·
                <button
                  type="button"
                  class="link-button"
                  @click="openFilePicker"
                >
                  browse
                </button>
              </span>
              <input
                ref="fileInput"
                type="file"
                accept="video/*,audio/*"
                class="file-input"
                @change="onFilePicked"
              />
            </div>

            <div v-if="selectedFileName" class="file-chip">
              <span>{{ selectedFileName }}</span>
              <button
                type="button"
                class="file-chip-clear"
                aria-label="Remove file"
                @click="clearFile"
              >
                ×
              </button>
            </div>
          </div>

          <button
            type="submit"
            class="primary-button"
            :disabled="!canSubmit"
          >
            Watch with Subs
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
            <div
              class="progress-fill"
              :style="{ width: `${progress}%` }"
            ></div>
          </div>
        </div>
      </section>

      <!-- READY / PLAYING -->
      <section
        v-else
        class="card card--player"
        :class="{ 'card--playing': stage === 'playing' }"
      >
        <div v-if="stage === 'ready'" class="progress-wrapper progress-wrapper--top">
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
          @play="onVideoPlay"
        ></video>

        <button class="ghost-button home-button" @click="goHome">
          ← Process another video
        </button>
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
  width: 56px;
  height: 56px;
  border-radius: 12px;
  object-fit: cover;
  box-shadow: var(--shadow-lg);
}

.title {
  font-size: 28px;
  font-weight: 800;
  margin: 0;
  letter-spacing: -0.02em;
}

.title-ellipsis {
  color: var(--text-muted);
  font-weight: 500;
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

.dropzone {
  border: 1.5px dashed var(--border);
  border-radius: var(--radius);
  padding: 20px;
  background: var(--bg-elev-2);
  transition:
    border-color 200ms ease,
    background 200ms ease;
}

.dropzone--over {
  border-color: var(--accent);
  background: var(--accent-soft);
}

.url-input {
  width: 100%;
  background: transparent;
  border: none;
  outline: none;
  color: var(--text);
  font-size: 16px;
  padding: 10px 4px;
}

.url-input::placeholder {
  color: var(--text-muted);
}

.dropzone-actions {
  margin-top: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.dropzone-hint {
  font-size: 13px;
  color: var(--text-muted);
}

.link-button {
  background: none;
  border: none;
  color: var(--accent);
  cursor: pointer;
  padding: 0;
  font-size: 13px;
  font-weight: 500;
}

.link-button:hover {
  color: var(--accent-hover);
  text-decoration: underline;
}

.file-input {
  display: none;
}

.file-chip {
  margin-top: 12px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--accent-soft);
  color: var(--text);
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 13px;
}

.file-chip-clear {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
  padding: 0;
}

.file-chip-clear:hover {
  color: var(--text);
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

.progress-wrapper--top {
  /* lives inside card--player at the top */
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
