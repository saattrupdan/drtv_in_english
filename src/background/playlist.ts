// Resolve DR's subtitle HLS playlist into a merged cue list.
//
// Each segment is its own WebVTT document with overlapping cues at
// the boundaries; we dedupe by (start, text). Segments are fetched in
// parallel — they're tiny, and DR's CDN is fast.

import { parseVtt } from "./vtt-parser.js";
import type { Cue } from "../shared/types.js";

export async function fetchCuesFromPlaylist(
  playlistUrl: string,
  signal: AbortSignal,
): Promise<Cue[]> {
  const res = await fetch(playlistUrl, { signal });
  if (!res.ok) throw new Error(`playlist fetch ${res.status}`);
  const text = await res.text();
  const segmentUris = parseM3u8Segments(text);
  if (segmentUris.length === 0) throw new Error("playlist had no segments");

  const base = new URL(playlistUrl);
  const segmentUrls = segmentUris.map((u) => new URL(u, base).toString());

  const results = await Promise.all(
    segmentUrls.map(async (u) => {
      const r = await fetch(u, { signal });
      if (!r.ok) throw new Error(`segment fetch ${r.status} for ${u}`);
      return r.text();
    }),
  );

  const merged: Cue[] = [];
  const seen = new Set<string>();
  for (const body of results) {
    for (const cue of parseVtt(body)) {
      const key = `${cue.start.toFixed(3)}|${cue.text}`;
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(cue);
    }
  }
  merged.sort((a, b) => a.start - b.start);
  return merged;
}

export async function fetchCuesFromSingleVtt(
  url: string,
  signal: AbortSignal,
): Promise<Cue[]> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`vtt fetch ${res.status}`);
  return parseVtt(await res.text());
}

function parseM3u8Segments(body: string): string[] {
  return body
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));
}
