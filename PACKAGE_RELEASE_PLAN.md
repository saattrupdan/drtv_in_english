# Package Release Plan — DRTV in English

This document walks through getting the extension live in both stores.

**Prerequisites:** The Phase 5 code changes are already committed — icons, privacy policy, tightened manifests, store listing copy, and permission justifications are all in the repo.

---

## Step 1: Host the privacy policy

The privacy policy lives at `privacy-policy.html` in the repo. Stores require a publicly accessible URL.

**Options:**
- **GitHub Pages** — push the repo (or a dedicated branch) to GitHub and enable Pages. URL: `https://dansmart.github.io/drtv_in_english/privacy-policy.html`
- **Any web host** — upload the HTML file somewhere.
- **Raw GitHub blob** — not ideal (stores sometimes reject `raw.githubusercontent.com` links), but functional.

**Action:** Pick a host, get a stable URL, note it down. You'll paste it into both store dashboards.

---

## Step 2: Take screenshots

Stores require screenshots at **1280×800**. You need three:

1. **Options page** — showing the provider selector (OpenAI / Anthropic / Custom) and API key input
2. **Three-way subtitle button** — the injected button on DR's player showing Dansk/English/Off
3. **English subs rendering** — a frame with English subtitles visible over the video

**How:**
- Go to `dr.dk/drtv` with the extension loaded (dev mode)
- Open options page → screenshot
- Play an episode, click the subtitle button → screenshot
- Show English subs on screen → screenshot

Resize each to exactly 1280×800 if needed.

---

## Step 3: Submit to Chrome Web Store

1. Go to [chrome.google.com/webstore/devconsole](https://chrome.google.com/webstore/devconsole)
2. Pay the **$5 one-time developer fee** (if you haven't already)
3. Click **"New item"**
4. Upload the **zipped extension** — run `npm run build`, then zip the `dist/chrome/` folder contents
5. Fill in:
   - **Title:** DRTV in English
   - **Description:** Use the long description from `store-submission/STORE-LISTING.md`
   - **Category:** Accessibility
   - **Privacy policy URL:** the URL from Step 1
   - **Screenshots:** upload the 3 images from Step 2
   - **Rating:** Adults — not suitable
6. Submit for review

Review typically takes 1–3 days.

---

## Step 4: Submit to Firefox Add-ons (AMO)

1. Go to [addons.mozilla.org/developers/addsubmit](https://addons.mozilla.org/developers/addsubmit)
2. Sign in with your Mozilla account
3. Upload the **zipped Firefox build** — run `npm run build`, then zip the `dist/firefox/` folder contents
4. Fill in:
   - **Name:** DRTV in English
   - **Summary:** Use the short description from `store-submission/STORE-LISTING.md`
   - **Description:** Use the long description
   - **Category:** Accessibility
   - **Privacy policy:** AMO doesn't require a URL but having one is good practice. Link to the same page.
5. Submit

AMO usually auto-approves unsigned extensions within minutes (Developer Edition / Nightly only). For regular Firefox, AMO signs it automatically.

---

## Step 5: Test the live install

Once both stores approve:

1. Open a **fresh browser profile** (no dev mode, no prior extension installs)
2. Install from the store
3. Go to `dr.dk/drtv` — confirm the extension icon appears
4. Open options, enter an API key (test with a free tier if possible)
5. Play a DRTV episode, click the subtitle button, verify English subs appear

---

## Checklist

- [ ] Privacy policy hosted at a stable URL
- [ ] 3 screenshots taken at 1280×800
- [ ] Chrome Web Store submitted ($5 fee)
- [ ] Firefox AMO submitted
- [ ] Fresh-profile install tested end-to-end
