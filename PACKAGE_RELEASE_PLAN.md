# Package Release Plan — DRTV in English

This document walks through getting the extension live in both stores.

**Prerequisites:** The Phase 5 code changes are already committed — icons, privacy policy, tightened manifests, store listing copy, and permission justifications are all in the repo.

**Privacy Policy URL:** `https://saattrupdan.github.io/drtv_in_english/privacy-policy.html`

## Package Directory Structure

All release artifacts are consolidated in `package/`:

```
package/
  chrome/                  # Chrome build output (for testing)
  firefox/                 # Firefox build output (for testing)
  zips/
    drtv-in-english-chrome-1.0.0.zip        # Upload to Chrome Web Store
    drtv-in-english-firefox-1.0.0.zip       # Upload to Firefox AMO
    drtv-in-english-source.zip              # Source package for Mozilla review
  assets/                   # Store screenshots (1280x800 JPEG)
    drtv-in-english-options-chrome.jpg
    drtv-in-english-subs-menu-chrome.jpg
    drtv-in-english-subs-chrome.jpg
    drtv-in-english-options-firefox.jpg
    drtv-in-english-subs-menu-firefox.jpg
    drtv-in-english-subs-firefox.jpg
  submission/
    STORE-LISTING.md                        # Store descriptions
    PERMISSION-JUSTIFICATIONS.md            # Firefox permission justifications
    SOURCE_SUBMISSION.md                    # Build instructions for Mozilla
```

---

## Step 1: Host the privacy policy ✅

**Done:** Privacy policy is live at `https://saattrupdan.github.io/drtv_in_english/privacy-policy.html`

Hosted on GitHub Pages from the `gh-pages` branch. Use this URL for both store dashboards.

---

## Step 2: Take screenshots ✅

Screenshots in `package/assets/`:

**For Chrome Web Store:**
1. `drtv-in-english-options-chrome.jpg` — Options page
2. `drtv-in-english-subs-menu-chrome.jpg` — Subtitle menu
3. `drtv-in-english-subs-chrome.jpg` — English subs

**For Firefox Add-ons:**
1. `drtv-in-english-options-firefox.jpg` — Options page
2. `drtv-in-english-subs-menu-firefox.jpg` — Subtitle menu
3. `drtv-in-english-subs-firefox.jpg` — English subs

All images are 1280×800 JPEGs, ready for upload.

---

## Step 3: Submit to Chrome Web Store

1. Go to [chrome.google.com/webstore/devconsole](https://chrome.google.com/webstore/devconsole)
2. Pay the **$5 one-time developer fee** (if you haven't already)
3. Click **"New item"**
4. Upload **`package/zips/drtv-in-english-chrome-0.1.0-dev.zip`**
5. Fill in:
   - **Title:** DRTV in English
   - **Description:** Use the long description from `package/submission/STORE-LISTING.md`
   - **Category:** Accessibility
   - **Privacy policy URL:** the URL from Step 1
   - **Screenshots:** upload the 3 images from `package/assets/`
   - **Rating:** Adults — not suitable
6. Submit for review

Review typically takes 1–3 days.

---

## Step 4: Submit to Firefox Add-ons (AMO)

1. Go to [addons.mozilla.org/developers/addsubmit](https://addons.mozilla.org/developers/addsubmit)
2. Sign in with your Mozilla account
3. Upload **`package/zips/drtv-in-english-firefox-0.1.0-dev.zip`**
4. Upload **`package/zips/drtv-in-english-source.zip`** when prompted for source code
5. Fill in:
   - **Name:** DRTV in English
   - **Summary:** Use the short description from `package/submission/STORE-LISTING.md`
   - **Description:** Use the long description
   - **Category:** Accessibility
   - **Privacy policy:** Link to `https://saattrupdan.github.io/drtv_in_english/privacy-policy.html`
   - **Reviewer notes:** Reference `package/submission/SOURCE_SUBMISSION.md` for build instructions
6. Submit

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

- [x] Privacy policy hosted at a stable URL
- [x] 3 screenshots taken at 1280×800
- [ ] Chrome Web Store submitted ($5 fee) — upload zips + 3 Chrome screenshots
- [ ] Firefox AMO submitted — upload zip + source + 3 Firefox screenshots
- [ ] Fresh-profile install tested end-to-end
