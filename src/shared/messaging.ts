// Episode id extraction shared by content + background.
//
// DRTV URLs come in two flavours:
//   /drtv/se/<slug>_<id>
//   /drtv/episode/<id>
// The trailing numeric (sometimes alphanumeric) segment after the final
// underscore is the canonical id used by DR's APIs.

export function extractEpisodeId(url: string): string | null {
  try {
    const u = new URL(url);
    if (!u.hostname.endsWith("dr.dk")) return null;
    const m =
      u.pathname.match(/\/drtv\/se\/[^/]*_([^/?#]+)/) ??
      u.pathname.match(/\/drtv\/episode\/([^/?#]+)/);
    return m ? (m[1] ?? null) : null;
  } catch {
    return null;
  }
}
