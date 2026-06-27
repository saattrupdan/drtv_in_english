// Runs at document_start in the isolated content-script world. Its only
// job is to bridge the page-world sniffer (content/inject.ts, which runs
// in the MAIN world) to the background service worker: it listens for
// the window messages inject.ts posts and relays subtitle URLs via
// chrome.runtime.sendMessage.
//
// Why this split:
//   - The MAIN-world script can wrap the page's fetch()/XHR but has no
//     access to chrome.runtime.
//   - This isolated-world script has chrome.runtime but can't see the
//     page's fetch()/XHR.
//   - DR's `script-src 'self' …` CSP blocks injecting an inline patch
//     from here (it's blocked in Chrome), so the patch must be a
//     declared MAIN-world content script instead.

(() => {
  const TAG = "[drtv-en/early]";

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

  console.log(TAG, "page-world bridge installed");
})();
