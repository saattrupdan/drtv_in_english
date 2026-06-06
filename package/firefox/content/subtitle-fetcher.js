"use strict";
(() => {
  // src/content/subtitle-fetcher.ts
  (() => {
    const TAG = "[drtv-en/fetcher]";
    chrome.runtime.onMessage.addListener(
      (msg, _sender, sendResponse) => {
        if (msg.type !== "fetch-url") return;
        fetch(msg.url).then((res) => {
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
          }
          return res.text();
        }).then(sendResponse).catch((err) => sendResponse({ error: err.message }));
        return true;
      }
    );
    console.log(TAG, "subtitle fetcher ready");
  })();
})();
//# sourceMappingURL=subtitle-fetcher.js.map
