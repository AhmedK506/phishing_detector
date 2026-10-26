# Phishing Detector — Project Summary

Last generated: 2026-07-20 (from a full read of the codebase, not from memory).

## 1. What this project is

An AI-assisted phishing-website detector with three parts:

1. **ML pipeline** (`model/`) — trains a Random Forest classifier on URL-derived features, plus a set of standalone data-prep/EDA/evaluation scripts.
2. **Flask backend** (`app.py`) — serves predictions via a REST API, combining three signals (URL model, live HTML inspection, visual brand-impersonation check) into one ensemble verdict. Also serves a live model-performance dashboard.
3. **Chrome extension (Manifest V3)** (`extension/`) — a popup UI to scan the current tab or a pasted URL, an auto-scan background worker that badges every tab, a content-script warning banner injected into risky pages, and a standalone Reports & History page.

Everything talks to the Flask server at `http://127.0.0.1:5000` — the extension has no functionality without it running locally.

---

## 2. Backend — `app.py`

Flask app with CORS enabled. On startup loads `model/phishing_model.pkl` and `model/feature_names.pkl` into globals (`model`, `feature_names`) — these are reused by every route that needs them, including the dashboard data route (no redundant reloading).

### Whitelist short-circuit
`WHITELIST` — ~35 hardcoded major/regional domains (Google, Facebook, Microsoft, plus Malaysian/SEA sites like `maybank2u.com.my`, `shopee.com.my`, `apu.edu.my`). `is_whitelisted(url)` uses `tldextract` to get the registrable domain and checks membership. If whitelisted, `/predict` returns an instant "Safe / Low / 0.0" result with `note: 'Whitelisted domain'`, skipping all analysis.

### Heuristic override — `path_impersonation_override(features)`
A conservative rule layered on top of the ML model. Fires **only** when three independent signals co-occur in the extracted URL features:
- `brand_in_path >= 1` (a brand name appears in the URL path)
- `suspicious_path_keywords >= 2` (≥2 phishing-kit words like verify/login/confirm in the path)
- `embedded_domain_in_path >= 1` (a domain-looking string is embedded in the path)

If triggered, the URL's ML phishing probability is floored at 85.0 (not slammed to 100 — kept "honest about being a heuristic").

### Ensemble scoring — `ensemble_decision(url_score, html_result, visual_result)`
Weighted sum: **URL 0.5, HTML 0.3, Visual 0.2**. If the HTML analysis failed to fetch the page, its score falls back to the URL score (not 0) so a failed fetch doesn't silently dilute the verdict. If visual analysis fails, its score is 0.
Thresholds: `final_score >= 60` → High/Phishing, `>= 35` → Medium/Suspicious, else Low/Safe.

### Routes
| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Health check JSON |
| `/predict` | POST | Body `{url, mode}`. `mode='url_only'` returns just the fast ML score (used by the extension's auto-scan). `mode='full'` (default) runs URL + HTML (`html_analyzer.analyze_html`) + Visual (`visual_analyzer.analyze_visual`) and returns the full ensemble breakdown. |
| `/static/<filename>` | GET | Serves chart PNGs from `static/` (confusion matrix, class distribution, binary balance — copies of the ones in `data/eda_charts/`). |
| `/dashboard` | GET | Returns `DASHBOARD_HTML` instantly — a dark-themed page with a spinner. No data is server-rendered. |
| `/dashboard/data` | GET | Does the actual work: loads `data/featurized_dataset.csv` + `data/featurized_tranco.csv`, concatenates + de-dupes, recreates the canonical `train_test_split(test_size=0.2, random_state=42, stratify=y)`, runs the loaded model on the test split, computes accuracy/precision/recall/f1/confusion matrix live via sklearn, and returns JSON (merged with some hardcoded fields: preprocessing counts, before/after `evaluation` numbers, `model` metadata). |

**Dashboard architecture**: `/dashboard` was deliberately split from `/dashboard/data` so the page loads instantly with a spinner, then `fetch('/dashboard/data')` client-side populates the DOM (metric cards, preprocessing/dataset tables, confusion-matrix grid, before/after bar comparisons for accuracy & recall) once the (slower, sklearn-computing) data arrives. Shows a red error message on fetch failure.

---

## 3. ML pipeline — `model/`

### 3.1 Feature extraction — `features.py`
`extract_features(url)` is the single source of truth for turning a raw URL string into a numeric feature vector. No live network calls — pure string/regex/`tldextract` analysis. It's imported directly by most training/eval scripts to featurize URLs on the fly.

Produces **38 features** per URL (note: `dashboard_data.json` claims `num_features: 35` — this is a stale/inconsistent value, not re-derived from the actual feature count):

- **Length features**: `url_length`, `domain_length`, `path_length`, `query_length`, `subdomain_length`
- **Character counts**: `num_dots`, `num_hyphens`, `num_underscores`, `num_slashes`, `num_question_marks`, `num_equal_signs`, `num_at_signs`, `num_percent`, `num_hash`, `num_digits`, `num_digits_in_domain`, `num_path_segments`
- **Structural flags**: `has_subdomain`, `is_https`, `domain_has_hyphen`, `domain_has_digits`, `has_ip` (regex-matches IPv4-as-hostname)
- **Keyword/brand signals**: `suspicious_keyword_count` (20-word list: login, verify, secure, banking, paypal, amazon...), `is_shortened` (bit.ly/tinyurl/etc.), `brand_in_subdomain` (8 brands), `brand_in_path` (15 brand/sensitive terms), `suspicious_path_keywords` (12 phishing-kit words), `embedded_domain_in_path` (regex for a second domain-like string inside the path — the signal used by the path-impersonation heuristic)
- **Ratios**: `digit_ratio`, `special_char_ratio`, `domain_consonant_ratio`
- **Obfuscation signals**: `has_long_hex_token` / `num_long_hex_tokens` (≥16-char hex strings, common in phishing-kit tracking IDs), `dots_in_path`, `max_consonant_run`
- **Entropy/linguistics**: `domain_entropy` (Shannon entropy of the domain string — random-looking domains score higher), `suspicious_tld` (15 cheap/abused TLDs: .tk, .xyz, .top, .click...), `is_normal_subdomain` (1 if subdomain is a common one like www/mail/login/shop — absence is a weak signal)

### 3.2 Live page analysis — `html_analyzer.py`
`analyze_html(url)` fetches the page (`requests`, spoofed Chrome UA, 8s timeout, `verify=False`) and parses with BeautifulSoup. On fetch failure, returns a default `risk_score=30, risk_level='Unknown'`. Otherwise builds an additive score (capped at 100):

| Signal | Points |
|---|---|
| Form action posts to an external domain | +30 |
| Hidden iframe(s) (`display:none`/`visibility:hidden`) | +20 |
| More than 3 iframes | +10 |
| External link ratio > 0.7 | +20 |
| ≥2 of `eval(`, `unescape(`, `document.write(`, `window.location`, `document.cookie`, `atob(`, `fromCharCode(` present | +15 |
| Password `<input>` present but page isn't HTTPS | +25 |
| Favicon hosted on an external domain | +10 |
| `<title>` contains login/verify/account/secure/update/confirm/banking | +10 |
| Right-click disabled (`contextmenu` + `return false` in JS) | +10 |

Risk level: ≥60 High, ≥30 Medium, else Low (same thresholds as the ensemble and visual analyzer).

### 3.3 Visual brand-impersonation check — `visual_analyzer.py`
Uses headless Selenium Chrome (via `webdriver_manager`) to screenshot the target page, then perceptual-hashes it (`imagehash.phash`) and compares (Hamming distance) against precomputed hashes of 8 real brand homepages (PayPal, Facebook, Google, Microsoft, Apple, Amazon, Netflix, Instagram) stored in `model/brand_hashes.pkl` (generated once via `generate_brand_hashes()`).

`similarity = max(0, 100 - distance * 1.5)`. If the best match's similarity is **> 80** AND the page's registered domain (via `tldextract`) does **not** match that brand's real registered domain → flags impersonation with `risk_score = min(100, similarity)` and an indicator string naming the impersonated brand and the actual hosting domain. If the domain does match the brand (i.e., it genuinely is that brand's site), score is forced to 0.

### 3.4 Data pipeline (run manually, in this order)

```
data/malicious_phish.csv (raw, 651,191 rows)
   │  preprocess.py — drops nulls/empty/malformed(no-dot) URLs, de-dupes by url,
   │  IQR-based outlier check on url_length (outliers KEPT — long URLs are a signal),
   │  writes 4 EDA charts to data/eda_charts/
   ▼
data/cleaned_dataset.csv (641,059 rows: url, type)
   │
   ├─ add_legit_data.py — appends top 30,000 data/tranco_top.csv domains as
   │  {'url': 'https://www.<domain>', 'type': 'benign'}, de-dupes
   │  → data/combined_dataset.csv (671,059 rows)
   │
   ├─ featurize_dataset.py — runs extract_features() over cleaned_dataset.csv
   │  → data/featurized_dataset.csv (641,059 rows × 38 features + url + label)
   │
   └─ featurize_tranco.py — same, over top-30,000 Tranco domains, label=0
      → data/featurized_tranco.csv (30,000 rows)
```

**Training scripts** (three parallel/legacy variants exist):
- **`train.py`** (current production path) — loads `combined_dataset.csv`, extracts features live, 80/20 stratified split (`random_state=42`), `RandomizedSearchCV` over `RandomForestClassifier(class_weight='balanced')` (grid: n_estimators [100,200,300], max_depth [10,20,30,None], min_samples_split [2,5,10], min_samples_leaf [1,2,4], max_features [sqrt,log2]; n_iter=10, cv=3, scoring='f1'). Saves `model/phishing_model.pkl` + `model/feature_names.pkl`. Recently updated to also `print(list(X.columns))` and `print(X.dtypes)` right after building the feature matrix, for debugging.
- **`train_from_features.py`** — same search, but trains from the pre-featurized CSVs instead of re-running `extract_features()` (faster iteration).
- **`train_cleaned_only.py`** — trains only on `cleaned_dataset.csv` (no Tranco), with a single fixed RF config (n_estimators=100, max_depth=30, min_samples_split=5, no search) → `phishing_model_cleaned_only.pkl`, used purely as a comparison baseline.

**Evaluation/analysis scripts** (all reproduce the identical `random_state=42, test_size=0.2, stratify=y` split so results line up with each other):
- **`confusion_matrix.py`** / **`cm_cleaned.py`** — confusion matrix + heatmap for the combined-data model vs the cleaned-only model respectively.
- **`save_metrics.py`** — the authoritative source of `data/metrics.json` (accuracy/precision/recall/f1/tn/fp/fn/tp/test_size), computed by re-splitting `combined_dataset.csv` and scoring `phishing_model.pkl`.
- **`threshold_optimization.py`** — sweeps classification thresholds 0.05–0.95 (step 0.05) on `predict_proba`, writes `data/threshold_analysis.csv` and 3 charts (precision-recall curve, metrics-vs-threshold, errors-vs-threshold).
- **`overfitting_check.py`** — compares train vs test accuracy/F1 gap (flags >0.10 as overfitting, >0.05 as mild), plus 5-fold `StratifiedKFold` cross-validation for stability.
- **`analyze_features.py`** — feature importances (`model.feature_importances_`, flags features contributing <0.005), false-positive/false-negative profiling (which feature averages differ most between correct and incorrect predictions), sample misclassified URLs, and a feature-correlation heatmap flagging pairs with `|r| >= 0.85` (redundancy check).
- **`evaluate.py`** — an independent "before vs after" comparison: 50 legit URLs sampled from Tranco ranks 30,001–60,000 (deliberately outside the top-30k used in training) + 50 phishing URLs from `data/verified_online.csv` (PhishTank). Runs both the OLD baseline model (`phishing_model_OLD.pkl`) and the current model through the same path-impersonation heuristic override used in `app.py`, and prints an old-vs-new comparison table. This is the source of the `evaluation` block manually copied into `dashboard_data.json`.
- **`label_consistency_check.py`**, **`check_tranco.py`**, **`preprocess_tranco.py`** — standalone data-quality/EDA scripts (conflicting labels for the same URL/domain, cross-source Tranco-vs-Kaggle label conflicts, Tranco null/malformed/duplicate checks, Tranco domain-length & TLD EDA). Console-report only, no output datasets.

**`build_dashboard_data.py`** — writes `data/dashboard_data.json` from a **hardcoded** Python dict (explicitly commented "these come from your actual runs — edit any that differ"), not computed live. It's manually kept in sync with the real outputs of `save_metrics.py` / `evaluate.py` / `preprocess.py`. `app.py`'s `/dashboard/data` route re-derives most of these numbers live instead of trusting this static file.

### 3.5 Legacy/leftover artifacts (present but not part of the current pipeline)
`phishing_model_OLD.pkl` / `feature_names_OLD.pkl` (baseline used only by `evaluate.py`), `phishing_model_v2.pkl`, `phishing_model_prefeatures.pkl`, `phishing_model_cleaned_only.pkl` and their `feature_names_*` counterparts, `model/tempCodeRunnerFile.py` (exact duplicate of `featurize_dataset.py`, an editor artifact).

### 3.6 Test fixtures — `test_samples/`
`fake_login.html` and `fake_payment.html` are hand-crafted synthetic phishing pages built to trip nearly every rule in `html_analyzer.analyze_html()` (external form action, hidden iframes, >3 iframes, suspicious JS, password field without HTTPS, external favicon, suspicious title, right-click disabled) — used for manually testing the HTML analyzer, not real captured phishing samples.

---

## 4. Chrome extension — `extension/` (Manifest V3)

| File | Role |
|---|---|
| `manifest.json` | MV3 manifest. Permissions: `activeTab`, `tabs`, `storage`, `scripting`. Host permissions: `http://127.0.0.1:5000/*` and `<all_urls>`. Background service worker `background.js`. Content script `content.js` on `<all_urls>` at `document_idle`. Popup `popup.html` with `default_icon: icon.png`. `report.html` declared under `web_accessible_resources` (needed since it's opened as its own tab via `chrome.runtime.getURL`). |
| `background.js` | Listens for `chrome.tabs.onUpdated` (page load complete) and `chrome.tabs.onActivated` (tab switch). `autoScan(tabId, url)` first checks `chrome.storage.local.autoScanEnabled` (default `true`) and bails out (clearing the badge) if disabled; skips non-`http` URLs; otherwise POSTs to `/predict` with `mode: 'url_only'` (fast, URL-only score) and sets the toolbar badge: red `!` (≥60% phishing prob), amber `?` (≥35%), green `✓` (safe). At ≥60% it also messages the content script to show the in-page warning banner. |
| `content.js` | Listens for a `showWarning` runtime message; injects a fixed-position red banner at the top of the page (with injected `@keyframes` for a slide-down animation) showing the risk %, with "Leave this site" (browser back, or google.com if no history) and "Dismiss" (slide back up + remove) buttons. |
| `popup.html` | Header → auto-scan toggle switch → mode toggle (Current Site / Paste URL) → URL box/input → Scan button → loading state (SVG circular progress ring + 4-step checklist: URL analysis/HTML inspection/Visual comparison/Ensemble scoring) → result section (SVG gauge, verdict card, 3 component score bars, indicators list, disclaimer) → error box → history section (Export/Clear buttons, scan list) → "View Model Dashboard" button → "View Reports & History" button. |
| `popup.js` | Wires all of the above: toggle syncs to/from `chrome.storage.local.autoScanEnabled`; `startLoader()`/`finishLoader()` animate the progress ring/steps; `setMode()` switches between scanning the current tab vs a pasted URL; the scan button POSTs to `/predict` (full mode) and calls `displayResult(data)`, which fills the gauge/verdict/component bars/indicators and calls `saveToHistory(data)`. `saveToHistory` stores a rich object per scan — `id` (`RPT-<timestamp suffix>`), `url`, `prediction`, `score`, `riskLevel`, `time`, `urlScore`, `htmlScore`, `visualScore`, `indicators` — into `chrome.storage.local.scanHistory`, capped at 50 entries (`MAX_HISTORY`). `renderHistory()` lists recent scans in the popup; Export builds/downloads a CSV; Clear wipes storage. Two `DOMContentLoaded` handlers open the dashboard (`http://127.0.0.1:5000/dashboard`) and the reports page (`chrome.runtime.getURL("report.html")`) in new tabs. |
| `styles.css` | Light-themed popup styling; includes the auto-scan toggle switch (`.switch`/`.slider`) and the shared `.dashboard-btn` style used by both footer buttons. |
| `report.html` / `report.js` | Standalone page (opened in its own tab, not the popup). Reads the same `chrome.storage.local.scanHistory`. Renders: a 3-stat summary (Total Scans / Phishing Detected / Safe Sites), a full history table, and a **formal threat report card** for every `Phishing`-predicted scan — report ID, timestamp, target URL, risk level, overall score, the 3 component scores, the indicator list, and a fixed recommendation block. "Clear All" wipes `scanHistory` after a `confirm()`. The inline `<script>` that used to live in `report.html` was extracted verbatim into `report.js` (referenced via `<script src="report.js">`). |
| `icon.png` | Extension/toolbar icon. |

---

## 5. Key numeric results (as of the last recorded run)

**Current production model, evaluated on the held-out 20% of `combined_dataset.csv`** (from `data/metrics.json`, matches `dashboard_data.json`'s `metrics`/`confusion` blocks exactly):
- Accuracy **97.01%**, Precision **90.75%**, Recall **92.36%**, F1 **91.55%**
- Confusion matrix: TN 108,462 · FP 2,215 · FN 1,797 · TP 21,738 (test_size 134,212)

**Dataset composition** (`dashboard_data.json`): raw 651,191 rows → 61 malformed removed, 10,071 duplicates removed → 641,059 cleaned rows; + 30,000 Tranco legit URLs → 671,059 total training rows. Class balance 66.78% safe / 33.22% malicious (≈2.01:1).

**Before/after held-out comparison** (`evaluate.py`-style, 50 legit + 50 phishing held-out URLs never used in training): accuracy 47.0% → 99.0%, recall 94.0% → 98.0%, false positives 50 → 0. This is the headline result showing the impact of adding Tranco data + the path-impersonation heuristic.

**Threshold sweep** (`data/threshold_analysis.csv`, 19 thresholds 0.05–0.95): best F1 (0.9319) at the **default 0.50** threshold; accuracy peaks slightly higher (0.9763) at 0.55. The default 0.5 cutoff is essentially already optimal — no change was warranted.

**Model config**: Random Forest, `RandomizedSearchCV`-tuned, `class_weight='balanced'` to handle the 2:1 class imbalance.

---

## 6. Known inconsistencies worth being aware of

- `features.py`'s `extract_features()` returns **38** feature keys by direct count, but `dashboard_data.json` (and its `num_features` field) says **35**. The dashboard number is a manually-curated value that wasn't updated when features were added — `app.py`'s live `/dashboard/data` route computes `num_features` from `len(feature_names)` instead, so the served dashboard is accurate even though the static JSON isn't.
- `build_dashboard_data.py` produces a **hardcoded** snapshot (`data/dashboard_data.json`) that is no longer what `/dashboard` actually serves — the Flask route now recomputes metrics live via `/dashboard/data`. The static JSON file and its generator script are effectively vestigial now.
- Several legacy model/feature-name `.pkl` pairs exist (`_OLD`, `_v2`, `_cleaned_only`, `_prefeatures`) from earlier iterations; only `phishing_model.pkl` / `feature_names.pkl` (no suffix) are current production.
