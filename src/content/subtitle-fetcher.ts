// Fetch subtitle playlists/segments in the page context (no CORS issues).
// Listens for messages from the background service worker and uses
// the page's fetch to retrieve subtitle data.

(() => {
  const TAG = "[drtv-en/fetcher]";

  // Listen for fetch requests from background
  chrome.runtime.onMessage.addListener(
    (msg: { type: "fetch-url"; url: string }, _sender, sendResponse) => {
      if (msg.type !== "fetch-url") return;

      // Fetch in page context (inherits dr.dk origin, no CORS issue)
      fetch(msg.url)
        .then((res) => {
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
          }
          return res.text();
        })
        .then(sendResponse)
        .catch((err) => sendResponse({ error: err.message }));

      return true; // Keep channel open for async response
    },
  );

  console.log(TAG, "subtitle fetcher ready");
})();
