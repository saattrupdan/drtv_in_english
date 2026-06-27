// Runs at document_start in the page's MAIN world (declared with
// "world": "MAIN" in both manifests). Wraps fetch() and
// XMLHttpRequest.open() so we observe every URL the DR player requests
// and relay subtitle URLs to the isolated-world content script
// (early.ts) via window.postMessage.
//
// Why a separate MAIN-world content script instead of injecting an
// inline <script> from early.ts: DR serves the page with a strict
// `script-src 'self' …` CSP that blocks inline script execution in
// Chrome. A declared MAIN-world content script runs in page context
// without tripping the CSP, so this works on both Chrome and Firefox.

(() => {
  // DR ships subs as an HLS subtitle playlist (.m3u8 with .vtt
  // segments). We surface both: the playlist URL lets us pull all
  // segments deterministically; segment URLs are a fallback if we
  // only see those.
  const VTT_RE =
    /\.vtt(\?|#|$)|\.webvtt(\?|#|$)|\/subtitles\/[^?#]*\.m3u8(\?|#|$)|\/master[^/]*\.m3u8(\?|#|$)/i;

  const post = (url: string) => {
    try {
      window.postMessage(
        { source: "drtv-en", type: "vtt-url", url },
        window.location.origin,
      );
    } catch (_err) {
      // Swallow — never break the page over telemetry.
    }
  };

  const origFetch = window.fetch.bind(window);
  window.fetch = function (input: RequestInfo | URL, init?: RequestInit) {
    try {
      const url =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : (input as Request).url;
      if (url && VTT_RE.test(url)) post(url);
    } catch (_err) {
      // ignore
    }
    return origFetch(input, init);
  };

  const origOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (
    this: XMLHttpRequest,
    method: string,
    url: string | URL,
    ...rest: unknown[]
  ): void {
    try {
      const s = typeof url === "string" ? url : url.toString();
      if (VTT_RE.test(s)) post(s);
    } catch (_err) {
      // ignore
    }
    return (origOpen as (...a: unknown[]) => void).apply(this, [
      method,
      url,
      ...rest,
    ]);
  } as typeof XMLHttpRequest.prototype.open;

  console.log("[drtv-en/inject] page-world VTT sniffer installed");
})();
