// Runs at document_start. Injects a page-world script that wraps
// fetch() and XMLHttpRequest.open() so we see every URL the DR player
// requests. When one looks like a WebVTT subtitle we relay it to the
// background.
//
// Why this rather than chrome.webRequest:
//   - Firefox MV3 treats all host_permissions as opt-in. Users must
//     grant access via about:addons before webRequest fires; an end-
//     user shouldn't have to know that.
//   - DR loads subs from a CDN (drod22s.akamaized.net), not from
//     dr.dk, so even with host_permissions granted we'd need a much
//     broader scope.
//   - Page-world monkey-patching needs zero extra permissions: the
//     content script is already allowed on dr.dk pages, and we
//     observe outgoing requests in-process.

(() => {
  const TAG = "[drtv-en/early]";

  const patch = () => {
    // This function is stringified and re-evaluated in the page world,
    // so it must be fully self-contained — no closure references.
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
  };

  const s = document.createElement("script");
  s.textContent = `(${patch.toString()})();`;
  (document.head || document.documentElement).appendChild(s);
  s.remove();

  window.addEventListener("message", (e: MessageEvent) => {
    if (e.source !== window) return;
    const data = e.data as { source?: string; type?: string; url?: string } | null;
    if (!data || data.source !== "drtv-en") return;
    if (data.type === "vtt-url" && data.url) {
      chrome.runtime.sendMessage({ type: "vtt-url", url: data.url }).catch(() => {
        // Background may be asleep at document_start; the next call wakes it.
      });
    }
  });

  console.log(TAG, "page-world VTT sniffer installed");
})();
