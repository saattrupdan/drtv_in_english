// IndexedDB-backed cache of completed translations.
//
// Keyed by (episodeId, sourceVttHash, provider, model) so that:
//   - if DR re-issues the Danish VTT for an episode, hash mismatch
//     forces a re-translate;
//   - switching providers/models gives you that provider's translation
//     without clobbering the previous one.
//
// chrome.storage.local has a 10MB quota; translations easily exceed
// that. IDB has no fixed quota.

import type { Cue } from "../shared/types.js";

const DB_NAME = "drtv-en-translations";
const STORE = "translations";
const VERSION = 1;

export interface CacheEntry {
  key: string; // composite key — see makeKey
  episodeId: string;
  sourceVttHash: string;
  provider: string;
  model: string;
  cuesJson: string;
  createdAt: number;
}

function makeKey(
  episodeId: string,
  hash: string,
  provider: string,
  model: string,
): string {
  return `${episodeId}|${hash}|${provider}|${model}`;
}

function openDb(): Promise<IDBDatabase> {
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

export async function getCachedCues(
  episodeId: string,
  hash: string,
  provider: string,
  model: string,
): Promise<Cue[] | null> {
  const db = await openDb();
  try {
    return await new Promise<Cue[] | null>((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(makeKey(episodeId, hash, provider, model));
      req.onsuccess = () => {
        const entry = req.result as CacheEntry | undefined;
        if (!entry) return resolve(null);
        try {
          resolve(JSON.parse(entry.cuesJson) as Cue[]);
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

export async function putCachedCues(
  episodeId: string,
  hash: string,
  provider: string,
  model: string,
  cues: Cue[],
): Promise<void> {
  const db = await openDb();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      const entry: CacheEntry = {
        key: makeKey(episodeId, hash, provider, model),
        episodeId,
        sourceVttHash: hash,
        provider,
        model,
        cuesJson: JSON.stringify(cues),
        createdAt: Date.now(),
      };
      tx.objectStore(STORE).put(entry);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}

export async function getCacheStats(): Promise<{
  entries: number;
  bytes: number;
}> {
  const db = await openDb();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).getAll();
      req.onsuccess = () => {
        const all = (req.result ?? []) as CacheEntry[];
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

export async function clearCache(): Promise<void> {
  const db = await openDb();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}

export async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const arr = Array.from(new Uint8Array(digest));
  return arr.map((b) => b.toString(16).padStart(2, "0")).join("");
}
