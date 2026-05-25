// Port of src/drtv_in_english/vtt.py — see the CRLF lesson in
// docs/extension-plan.md §"Lessons from prior iterations". Normalising
// \r\n -> \n before the cue regex matches is load-bearing; without it
// the greedy text capture swallows everything to EOF and you get a
// single cue containing the whole transcript.

import type { Cue } from "../shared/types.js";

const CUE = new RegExp(
  String.raw`(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})[^\n]*\n` +
    String.raw`((?:.+\n?)+?)(?=\n\s*\n|\n[^\n]*-->|$)`,
  "gm",
);

const TAG = /<[^>]+>/g;

export function parseVtt(raw: string): Cue[] {
  const text = raw.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const out: Cue[] = [];
  for (const m of text.matchAll(CUE)) {
    const start = parseTs(m[1]!);
    const end = parseTs(m[2]!);
    const lines = (m[3] ?? "")
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    const cleaned = lines.join(" ").replace(TAG, "").trim();
    if (!cleaned) continue;
    out.push({ start, end, text: cleaned });
  }
  return out;
}

function parseTs(s: string): number {
  const [h, mi, rest] = s.split(":");
  const [sec, ms] = (rest ?? "0.000").split(".");
  return (
    Number(h) * 3600 +
    Number(mi) * 60 +
    Number(sec) +
    Number(ms ?? "0") / 1000
  );
}
