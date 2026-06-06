// CORS proxy for LLM provider requests.
// Chrome MV3: uses declarativeNetRequest to modify response headers.
// Firefox MV3: uses webRequest with blocking (still supported).
//
// Accepts all origins so users can configure any LLM provider without CORS issues.

const CORS_PROXY_ORIGINS = ["<all_urls>"];

const CORS_HEADERS = [
  { name: "Access-Control-Allow-Origin", value: "*" },
  { name: "Access-Control-Allow-Methods", value: "GET, POST, OPTIONS" },
  { name: "Access-Control-Allow-Headers", value: "Authorization, Content-Type" },
];

export async function enableCorsProxy(): Promise<void> {
  const isFirefox = typeof browser !== "undefined";

  if (isFirefox) {
    // Firefox: use webRequest API (still works in MV3)
    enableCorsProxyFirefox();
  } else {
    // Chrome MV3: use declarativeNetRequest
    await enableCorsProxyChrome();
  }
}

function enableCorsProxyFirefox(): void {
  // Add CORS headers to ALL responses (for any LLM provider the user configures)
  chrome.webRequest.onHeadersReceived.addListener(
    (details) => {
      if (!details.responseHeaders) return {};

      // Filter out existing CORS headers
      const filtered = details.responseHeaders.filter(
        (h) => !h.name?.toLowerCase().startsWith("access-control-"),
      );

      // Add our CORS headers
      filtered.push(...CORS_HEADERS);

      return { responseHeaders: filtered };
    },
    {
      urls: ["<all_urls>"],
      types: ["xmlhttprequest"],
    },
    ["blocking", "responseHeaders"] as chrome.webRequest.OnHeadersReceivedOptions,
  );

  console.log("[drtv-en/bg] CORS proxy enabled via webRequest (Firefox)");
}

async function enableCorsProxyChrome(): Promise<void> {
  const rules: chrome.declarativeNetRequest.Rule[] = [
    {
      id: 1,
      priority: 1,
      action: {
        type: "modifyHeaders" as const,
        responseHeaders: CORS_HEADERS.map((h) => ({
          header: h.name,
          operation: "set" as const,
          value: h.value,
        })),
      },
      condition: {
        urlFilter: "*://*/*",
        resourceTypes: ["xmlhttprequest"],
      },
    },
  ];

  try {
    const existing = await chrome.declarativeNetRequest.getDynamicRules();
    if (existing.length > 0) {
      await chrome.declarativeNetRequest.updateDynamicRules({
        removeRuleIds: existing.map((r) => r.id),
      });
    }
    await chrome.declarativeNetRequest.updateDynamicRules({ addRules: rules });
    console.log("[drtv-en/bg] CORS proxy enabled via declarativeNetRequest (Chrome)");
  } catch (err) {
    console.warn("[drtv-en/bg] declarativeNetRequest failed, falling back to no-op", err);
  }
}
