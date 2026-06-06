"use strict";
(() => {
  // src/content/early.ts
  (() => {
    const TAG = "[drtv-en/early]";
    const patch = () => {
      const VTT_RE = /\.vtt(\?|#|$)|\.webvtt(\?|#|$)|\/subtitles\/[^?#]*\.m3u8(\?|#|$)|\/master[^/]*\.m3u8(\?|#|$)/i;
      const post = (url) => {
        try {
          window.postMessage(
            { source: "drtv-en", type: "vtt-url", url },
            window.location.origin
          );
        } catch (_err) {
        }
      };
      const origFetch = window.fetch.bind(window);
      window.fetch = function(input, init) {
        try {
          const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
          if (url && VTT_RE.test(url)) post(url);
        } catch (_err) {
        }
        return origFetch(input, init);
      };
      const origOpen = XMLHttpRequest.prototype.open;
      XMLHttpRequest.prototype.open = function(method, url, ...rest) {
        try {
          const s2 = typeof url === "string" ? url : url.toString();
          if (VTT_RE.test(s2)) post(s2);
        } catch (_err) {
        }
        return origOpen.apply(this, [
          method,
          url,
          ...rest
        ]);
      };
    };
    const s = document.createElement("script");
    s.textContent = `(${patch.toString()})();`;
    (document.head || document.documentElement).appendChild(s);
    s.remove();
    window.addEventListener("message", (e) => {
      if (e.source !== window) return;
      const data = e.data;
      if (!data || data.source !== "drtv-en") return;
      if (data.type === "vtt-url" && data.url) {
        chrome.runtime.sendMessage({ type: "vtt-url", url: data.url }).catch(() => {
        });
      }
    });
    console.log(TAG, "page-world VTT sniffer installed");
  })();
})();
//# sourceMappingURL=early.js.map
