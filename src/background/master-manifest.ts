// Parse DR's HLS master manifest and pick the comprehensive Danish
// subtitle track. Per-episode, DR exposes two LANGUAGE="da" tracks:
//
//   NAME="Fremmedsprogstekster"     DEFAULT=YES   sparse, foreign-only
//   NAME="Dansk"                    DEFAULT=NO    full dialogue
//
// We always want "Dansk". This file does just that picking.

export interface SubtitleTrack {
  uri: string;
  name: string;
  language: string;
  isDefault: boolean;
}

export function parseSubtitleTracks(masterText: string): SubtitleTrack[] {
  const tracks: SubtitleTrack[] = [];
  for (const line of masterText.split(/\r?\n/)) {
    if (!line.startsWith("#EXT-X-MEDIA")) continue;
    const attrs = parseAttrs(line);
    if (attrs.TYPE !== "SUBTITLES") continue;
    tracks.push({
      uri: attrs.URI ?? "",
      name: attrs.NAME ?? "",
      language: attrs.LANGUAGE ?? "",
      isDefault: attrs.DEFAULT === "YES",
    });
  }
  return tracks;
}

export function pickDanishTrack(
  tracks: SubtitleTrack[],
): SubtitleTrack | undefined {
  const da = tracks.filter((t) => /^da\b/i.test(t.language));
  if (da.length === 0) return undefined;

  // Preferred: NAME matches "Dansk" and not "Fremmed".
  const named = da.find(
    (t) => /dansk/i.test(t.name) && !/fremmed/i.test(t.name),
  );
  if (named) return named;

  // Next: the non-default track (DR marks the foreign-only one default).
  const nonDefault = da.find((t) => !t.isDefault);
  if (nonDefault) return nonDefault;

  return da[0];
}

function parseAttrs(line: string): Record<string, string> {
  const out: Record<string, string> = {};
  const colon = line.indexOf(":");
  const body = colon >= 0 ? line.slice(colon + 1) : line;
  const re = /([A-Z0-9-]+)=(?:"([^"]*)"|([^,]*))/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(body)) !== null) {
    const key = m[1]!;
    out[key] = m[2] ?? m[3] ?? "";
  }
  return out;
}
