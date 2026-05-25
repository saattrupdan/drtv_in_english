<script setup lang="ts">
import Hls from "hls.js";
import { computed, nextTick, onBeforeUnmount, ref, useTemplateRef } from "vue";

type Stage = "idle" | "browsing" | "preparing" | "playing" | "error";
type InputMode = "search" | "url";

interface PrepareResponse {
  job_id: string;
  title: string;
  hls_url: string;
  cue_count: number;
}

interface CueEvent {
  index: number;
  start: number;
  end: number;
  text: string;
  done: boolean;
}

interface SearchItem {
  title: string;
  subtitle: string;
  description: string;
  image: string;
  path: string;
  url: string;
  kind: "series" | "playable";
}

interface Episode {
  title: string;
  subtitle: string;
  image: string;
  url: string;
  episode_number: number | null;
  season_number: number | null;
}

function episodeLabel(ep: Episode): string {
  const s = ep.season_number;
  const e = ep.episode_number;
  if (s != null && e != null) return `S${s}:E${e}`;
  if (e != null) return `Episode ${e}`;
  return "";
}

const stage = ref<Stage>("idle");
const inputMode = ref<InputMode>("search");
const url = ref("");
const searchQuery = ref("");
const searchResults = ref<{ series: SearchItem[]; playable: SearchItem[] }>({
  series: [],
  playable: [],
});
const isSearching = ref(false);
const selectedSeriesTitle = ref("");
const episodes = ref<Episode[]>([]);
const isLoadingEpisodes = ref(false);
const errorMessage = ref<string | null>(null);
const prepared = ref<PrepareResponse | null>(null);
const firstCueReady = ref(false);
const isPlaying = ref(false);

const videoEl = useTemplateRef<HTMLVideoElement>("videoEl");

let hls: Hls | null = null;
let prepareAbort: AbortController | null = null;
let translateAbort: AbortController | null = null;
let searchAbort: AbortController | null = null;
let searchTimer: number | null = null;
let englishTrack: TextTrack | null = null;
let pendingCues: CueEvent[] = [];

const canSubmit = computed(() => url.value.trim().length > 0);
const hasSearchResults = computed(
  () =>
    searchResults.value.series.length > 0 ||
    searchResults.value.playable.length > 0,
);

async function startPreparing(targetUrl?: string) {
  const sourceUrl = (targetUrl ?? url.value).trim();
  if (!sourceUrl) return;
  stage.value = "preparing";
  errorMessage.value = null;
  prepared.value = null;
  firstCueReady.value = false;
  isPlaying.value = false;
  pendingCues = [];

  prepareAbort = new AbortController();
  try {
    const response = await fetch("/api/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: sourceUrl, language: "en" }),
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

  video.muted = false;
  video.volume = 1.0;

  englishTrack = video.addTextTrack("subtitles", "English", "en");
  englishTrack.mode = "showing";
  for (const event of pendingCues) {
    englishTrack.addCue(new VTTCue(event.start, event.end, event.text));
  }
  pendingCues = [];

}

function startPlayback() {
  const video = videoEl.value;
  if (!video) return;
  video.muted = false;
  isPlaying.value = true;
  void video.play().catch(() => {
    isPlaying.value = false;
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
  if (event.done) return;

  if (englishTrack) {
    englishTrack.addCue(new VTTCue(event.start, event.end, event.text));
  } else {
    pendingCues.push(event);
  }

  if (!firstCueReady.value) {
    firstCueReady.value = true;
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

function onSearchInput() {
  if (searchTimer !== null) {
    clearTimeout(searchTimer);
    searchTimer = null;
  }
  const q = searchQuery.value.trim();
  if (!q) {
    if (searchAbort) searchAbort.abort();
    searchResults.value = { series: [], playable: [] };
    isSearching.value = false;
    return;
  }
  searchTimer = window.setTimeout(() => {
    void runSearch(q);
  }, 250);
}

async function runSearch(q: string) {
  if (searchAbort) searchAbort.abort();
  searchAbort = new AbortController();
  isSearching.value = true;
  try {
    const response = await fetch(
      `/api/search?q=${encodeURIComponent(q)}`,
      { signal: searchAbort.signal },
    );
    if (!response.ok) throw new Error(`Search failed (${response.status})`);
    searchResults.value = (await response.json()) as {
      series: SearchItem[];
      playable: SearchItem[];
    };
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    searchResults.value = { series: [], playable: [] };
  } finally {
    isSearching.value = false;
  }
}

async function selectItem(item: SearchItem) {
  if (item.kind === "playable") {
    void startPreparing(item.url);
    return;
  }
  selectedSeriesTitle.value = item.title;
  episodes.value = [];
  isLoadingEpisodes.value = true;
  stage.value = "browsing";
  try {
    const response = await fetch(
      `/api/episodes?path=${encodeURIComponent(item.path)}`,
    );
    if (!response.ok) {
      const detail = await safeDetail(response);
      throw new Error(detail ?? `Failed to load episodes (${response.status})`);
    }
    const data = (await response.json()) as { title: string; episodes: Episode[] };
    selectedSeriesTitle.value = data.title || item.title;
    episodes.value = data.episodes;
  } catch (err) {
    stage.value = "error";
    errorMessage.value = (err as Error).message || "Failed to load episodes.";
  } finally {
    isLoadingEpisodes.value = false;
  }
}

function selectEpisode(ep: Episode) {
  void startPreparing(ep.url);
}

function backToSearch() {
  stage.value = "idle";
  episodes.value = [];
  selectedSeriesTitle.value = "";
}

function showUrlInput() {
  inputMode.value = "url";
}

function showSearchInput() {
  inputMode.value = "search";
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
  firstCueReady.value = false;
  isPlaying.value = false;
  pendingCues = [];
  episodes.value = [];
  selectedSeriesTitle.value = "";
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
        <template v-if="inputMode === 'search'">
          <p class="lede">Search DRTV and watch with English subtitles.</p>

          <input
            v-model="searchQuery"
            type="text"
            class="url-input"
            placeholder="Search shows, films, episodes…"
            autofocus
            @input="onSearchInput"
          />

          <div v-if="isSearching" class="search-status">
            <div class="spinner spinner--sm" aria-hidden="true"></div>
            <span>Searching…</span>
          </div>

          <div v-if="hasSearchResults" class="results">
            <div
              v-for="item in [
                ...searchResults.series,
                ...searchResults.playable,
              ]"
              :key="item.kind + ':' + item.path"
              class="result"
              role="button"
              tabindex="0"
              @click="selectItem(item)"
              @keydown.enter.prevent="selectItem(item)"
            >
              <img
                v-if="item.image"
                :src="item.image"
                :alt="item.title"
                class="result-image"
                loading="lazy"
              />
              <div class="result-text">
                <div class="result-title">
                  {{ item.title }}
                  <span v-if="item.kind === 'series'" class="result-badge">
                    Series
                  </span>
                </div>
                <div v-if="item.subtitle" class="result-subtitle">
                  {{ item.subtitle }}
                </div>
                <div v-if="item.description" class="result-desc">
                  {{ item.description }}
                </div>
              </div>
            </div>
          </div>

          <button class="link-button" @click="showUrlInput">
            or paste a URL manually
          </button>
        </template>

        <template v-else>
          <p class="lede">Paste a DRTV URL and watch with English subtitles.</p>
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
          <button class="link-button" @click="showSearchInput">
            ← back to search
          </button>
        </template>
      </section>

      <section v-else-if="stage === 'browsing'" class="card">
        <button class="link-button back-link" @click="backToSearch">
          ← back to search
        </button>
        <h2 class="series-title">{{ selectedSeriesTitle }}</h2>

        <div v-if="isLoadingEpisodes" class="search-status">
          <div class="spinner spinner--sm" aria-hidden="true"></div>
          <span>Loading episodes…</span>
        </div>

        <div v-else-if="episodes.length === 0" class="search-status">
          <span>No episodes found.</span>
        </div>

        <div v-else class="results">
          <div
            v-for="ep in episodes"
            :key="ep.url"
            class="result"
            role="button"
            tabindex="0"
            @click="selectEpisode(ep)"
            @keydown.enter.prevent="selectEpisode(ep)"
          >
            <img
              v-if="ep.image"
              :src="ep.image"
              :alt="ep.title"
              class="result-image"
              loading="lazy"
            />
            <div class="result-text">
              <div class="result-title">
                <span v-if="episodeLabel(ep)" class="episode-tag">
                  {{ episodeLabel(ep) }}
                </span>
                {{ ep.title }}
              </div>
              <div v-if="ep.subtitle" class="result-desc">
                {{ ep.subtitle }}
              </div>
            </div>
          </div>
        </div>
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
        <div class="video-wrapper">
          <video
            v-show="isPlaying"
            ref="videoEl"
            class="video"
            controls
            controlslist="nodownload"
            playsinline
          ></video>
          <div v-if="!isPlaying" class="video-splash">
            <template v-if="!firstCueReady">
              <div class="spinner" aria-hidden="true"></div>
              <span class="preparing-label">Translating subtitles…</span>
            </template>
            <button v-else class="play-button" @click="startPlayback">
              <svg viewBox="0 0 24 24" width="32" height="32" aria-hidden="true">
                <path d="M8 5v14l11-7z" fill="currentColor" />
              </svg>
              <span>Play with English subs</span>
            </button>
          </div>
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

.video-wrapper {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  border-radius: var(--radius);
  overflow: hidden;
}

.video-splash {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  background: #000;
  color: var(--text-muted);
}

.video {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #000;
  border-radius: var(--radius);
  display: block;
}

.play-button {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  background: var(--accent);
  color: white;
  border: none;
  padding: 14px 22px;
  border-radius: var(--radius);
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: background 150ms ease;
}

.play-button:hover {
  background: var(--accent-hover);
}

.search-status {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
  font-size: 13px;
  color: var(--text-muted);
}

.results {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 20px;
  max-height: 540px;
  overflow-y: auto;
}

.result {
  display: flex;
  gap: 14px;
  padding: 10px;
  border-radius: 10px;
  border: 1px solid transparent;
  cursor: pointer;
  transition:
    background 150ms ease,
    border-color 150ms ease;
}

.result:hover,
.result:focus-visible {
  background: var(--bg-elev-2);
  border-color: var(--border);
  outline: none;
}

.result-image {
  width: 140px;
  height: 78px;
  object-fit: cover;
  border-radius: 6px;
  background: var(--bg-elev-2);
  flex-shrink: 0;
}

.result-text {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.result-title {
  font-size: 15px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}

.episode-tag {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  padding: 2px 6px;
  background: var(--bg-elev-2);
  color: var(--text-muted);
  border-radius: 4px;
  font-variant-numeric: tabular-nums;
}

.result-badge {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 2px 6px;
  background: var(--accent);
  color: white;
  border-radius: 4px;
}

.result-subtitle {
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.result-desc {
  font-size: 13px;
  color: var(--text-muted);
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.link-button {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 13px;
  padding: 8px 0;
  margin-top: 12px;
  text-decoration: underline;
  text-underline-offset: 3px;
  align-self: flex-start;
}

.link-button:hover {
  color: var(--text);
}

.back-link {
  margin-top: 0;
  margin-bottom: 16px;
}

.series-title {
  font-size: 22px;
  font-weight: 700;
  margin: 0 0 8px;
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
