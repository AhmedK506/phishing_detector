const API_URL = "http://127.0.0.1:5000/predict";

// ── Auto-scan on/off toggle ───────────────────────────────
const autoscanToggle = document.getElementById("autoscan-toggle");

// Load saved state (default ON)
chrome.storage.local.get({ autoScanEnabled: true }, (data) => {
  autoscanToggle.checked = data.autoScanEnabled;
});

// Save when changed
autoscanToggle.addEventListener("change", () => {
  chrome.storage.local.set({ autoScanEnabled: autoscanToggle.checked });
  // If turned off, clear the badge on the current tab
  if (!autoscanToggle.checked) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) chrome.action.setBadgeText({ tabId: tabs[0].id, text: "" });
    });
  }
});

let progressTimer = null;

const loadingStages = [
  { pct: 25, msg: "Analyzing URL structure...", step: 1 },
  { pct: 55, msg: "Inspecting page HTML...",     step: 2 },
  { pct: 80, msg: "Comparing visual layout...",  step: 3 },
  { pct: 95, msg: "Calculating risk score...",   step: 4 },
];

function startLoader() {
  const ring = document.getElementById("prog-ring");
  const pctText = document.getElementById("prog-pct");
  const msg = document.getElementById("loading-msg");
  const circumference = 176;

  // reset steps
  [1,2,3,4].forEach(n => {
    document.getElementById("step-" + n).classList.remove("active", "done");
  });
  msg.textContent = "Initializing scan...";
  pctText.textContent = "0%";
  ring.setAttribute("stroke-dashoffset", circumference);

  let stageIndex = 0;
  let currentPct = 0;

  progressTimer = setInterval(() => {
    if (stageIndex < loadingStages.length) {
      const stage = loadingStages[stageIndex];

      // creep toward this stage's target percentage
      if (currentPct < stage.pct) {
        currentPct += 1;
        pctText.textContent = currentPct + "%";
        ring.setAttribute("stroke-dashoffset", circumference - (circumference * currentPct / 100));
      } else {
        // reached the stage: update message + steps
        msg.textContent = stage.msg;
        for (let n = 1; n <= 4; n++) {
          const el = document.getElementById("step-" + n);
          if (n < stage.step) { el.classList.add("done"); el.classList.remove("active"); }
          else if (n === stage.step) { el.classList.add("active"); el.classList.remove("done"); }
        }
        stageIndex++;
      }
    }
  }, 60);
}

function finishLoader() {
  clearInterval(progressTimer);
  const ring = document.getElementById("prog-ring");
  const pctText = document.getElementById("prog-pct");
  ring.setAttribute("stroke-dashoffset", 0);
  pctText.textContent = "100%";
  [1,2,3,4].forEach(n => {
    const el = document.getElementById("step-" + n);
    el.classList.remove("active"); el.classList.add("done");
  });
}

let currentTabUrl = "";
let mode = "current";

chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  if (tabs[0] && tabs[0].url) {
    currentTabUrl = tabs[0].url;
    document.getElementById("current-url").textContent = currentTabUrl;
  } else {
    document.getElementById("current-url").textContent = "No URL found";
  }
});

const modeCurrentBtn = document.getElementById("mode-current");
const modePasteBtn = document.getElementById("mode-paste");
const urlBox = document.getElementById("url-box");
const urlInput = document.getElementById("url-input");
const scanBtn = document.getElementById("scan-btn");

function setMode(newMode) {
  mode = newMode;
  document.getElementById("result").classList.add("hidden");
  document.getElementById("error").classList.add("hidden");
  if (mode === "current") {
    modeCurrentBtn.classList.add("active");
    modePasteBtn.classList.remove("active");
    urlBox.classList.remove("hidden");
    urlInput.classList.add("hidden");
    scanBtn.textContent = "Scan This Site";
  } else {
    modePasteBtn.classList.add("active");
    modeCurrentBtn.classList.remove("active");
    urlBox.classList.add("hidden");
    urlInput.classList.remove("hidden");
    urlInput.value = "";
    scanBtn.textContent = "Scan This URL";
    urlInput.focus();
  }
}
modeCurrentBtn.addEventListener("click", () => setMode("current"));
modePasteBtn.addEventListener("click", () => setMode("paste"));

scanBtn.addEventListener("click", async () => {
  const loading = document.getElementById("loading");
  const result = document.getElementById("result");
  const errorBox = document.getElementById("error");

  let urlToScan = mode === "current" ? currentTabUrl : urlInput.value.trim();

  result.classList.add("hidden");
  errorBox.classList.add("hidden");

  if (urlToScan === "") {
    errorBox.textContent = "Please paste a URL first.";
    errorBox.classList.remove("hidden");
    return;
  }
  if (urlToScan.startsWith("chrome://") || urlToScan.startsWith("edge://")) {
    errorBox.textContent = "Cannot scan browser internal pages.";
    errorBox.classList.remove("hidden");
    return;
  }
  if (!urlToScan.startsWith("http://") && !urlToScan.startsWith("https://")) {
    urlToScan = "http://" + urlToScan;
  }

  loading.classList.remove("hidden");
  scanBtn.disabled = true;
  startLoader();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: urlToScan })
    });
    if (!response.ok) throw new Error("Server error");
    const data = await response.json();
    finishLoader();
    setTimeout(() => displayResult(data), 300);
  } catch (err) {
    clearInterval(progressTimer);
    loading.classList.add("hidden");
    errorBox.textContent = "Could not connect to detector. Make sure the Flask server is running.";
    errorBox.classList.remove("hidden");
  } finally {
    scanBtn.disabled = false;
  }
});

function colorFor(score) {
  if (score >= 60) return "#ef4444";   // red
  if (score >= 35) return "#f59e0b";   // amber
  return "#22c55e";                    // green
}

function displayResult(data) {
  document.getElementById("loading").classList.add("hidden");
  document.getElementById("result").classList.remove("hidden");

  const prediction = data.prediction || "Unknown";
  const score = Math.round(data.final_score ?? 0);
  const riskLevel = data.risk_level ?? "N/A";

  // Gauge
  const circumference = 204; // length of the arc path
  const offset = circumference - (circumference * score / 100);
  const gaugeFill = document.getElementById("gauge-fill");
  gaugeFill.setAttribute("stroke-dashoffset", offset);
  gaugeFill.setAttribute("stroke", colorFor(score));
  document.getElementById("gauge-score").textContent = score;
  document.getElementById("gauge-label").textContent = (riskLevel + " RISK").toUpperCase();

  // Verdict card (icon + title + subtitle)
  const card = document.getElementById("verdict-card");
  const icon = document.getElementById("verdict-icon");
  const vtext = document.getElementById("verdict-text");
  const vsub = document.getElementById("verdict-sub");

  card.className = "verdict-card " + prediction.toLowerCase();

  if (prediction.toLowerCase() === "safe") {
    icon.textContent = "✅";
    vtext.textContent = "Looks Safe";
  } else if (prediction.toLowerCase() === "suspicious") {
    icon.textContent = "⚠️";
    vtext.textContent = "Suspicious";
  } else {
    icon.textContent = "⚠️";
    vtext.textContent = "Phishing Detected";
  }
  vsub.textContent = "Risk score " + score + " / 100";

  // Components
  const comps = data.components || {};
  const urlScore = comps.url_analysis?.phishing_probability ?? 0;
  const htmlScore = comps.html_analysis?.risk_score ?? 0;
  const visualScore = comps.visual_analysis?.risk_score ?? 0;

  setBar("url", urlScore, urlScore);
  setBar("html", htmlScore, htmlScore);
  setBar("visual", visualScore, visualScore);

  // Indicators
  const allIndicators = [
    ...(comps.html_analysis?.indicators || []),
    ...(comps.visual_analysis?.indicators || [])
  ];
  const box = document.getElementById("indicators");
  if (allIndicators.length > 0) {
    let html = "<h3>Why was this flagged?</h3>";
    allIndicators.forEach(ind => { html += `<div class="indicator-item">⚠️ ${ind}</div>`; });
    box.innerHTML = html;
  } else {
    box.innerHTML = "<h3>No suspicious indicators found</h3>";
  }

  // Save this scan to history
  saveToHistory(data);
}

function setBar(name, displayValue, scoreForColor) {
  const label = document.getElementById(name + "-score");
  const bar = document.getElementById(name + "-bar");
  label.textContent = name === "url" ? displayValue + "%" : displayValue;
  bar.style.width = Math.min(scoreForColor, 100) + "%";
  bar.style.background = colorFor(scoreForColor);
}

// ── Save full scan result to history ──────────────────────
const MAX_HISTORY = 50;

function saveToHistory(data) {
  chrome.storage.local.get({ scanHistory: [] }, (stored) => {
    const history = stored.scanHistory;
    const comps = data.components || {};
    history.unshift({
      id: "RPT-" + Date.now().toString().slice(-8),
      url: data.url,
      prediction: data.prediction,
      score: Math.round(data.final_score ?? 0),
      riskLevel: data.risk_level ?? "N/A",
      time: Date.now(),
      urlScore: comps.url_analysis?.phishing_probability ?? 0,
      htmlScore: comps.html_analysis?.risk_score ?? 0,
      visualScore: comps.visual_analysis?.risk_score ?? 0,
      indicators: [
        ...(comps.html_analysis?.indicators || []),
        ...(comps.visual_analysis?.indicators || [])
      ]
    });
    const trimmed = history.slice(0, MAX_HISTORY);
    chrome.storage.local.set({ scanHistory: trimmed }, renderHistory);
  });
}

function renderHistory() {
  chrome.storage.local.get({ scanHistory: [] }, (data) => {
    const list = document.getElementById("history-list");
    const history = data.scanHistory;

    if (history.length === 0) {
      list.innerHTML = '<p class="history-empty">No scans yet.</p>';
      return;
    }

    list.innerHTML = "";
    history.forEach(item => {
      const color = item.score >= 60 ? "#ef4444"
                  : item.score >= 35 ? "#f59e0b" : "#22c55e";
      const row = document.createElement("div");
      row.className = "history-item";
      row.innerHTML = `
        <span class="history-badge" style="background:${color}"></span>
        <span class="history-url" title="${item.url}">${item.url}</span>
        <span class="history-score">${item.score}</span>
      `;
      list.appendChild(row);
    });
  });
}

document.getElementById("clear-history").addEventListener("click", () => {
  chrome.storage.local.set({ scanHistory: [] }, renderHistory);
});

// render history when popup opens
renderHistory();

// ── Export scan history as CSV (opens in Excel) ───────────
document.getElementById("export-history").addEventListener("click", () => {
  chrome.storage.local.get({ scanHistory: [] }, (data) => {
    const history = data.scanHistory;

    if (history.length === 0) {
      alert("No scans to export yet.");
      return;
    }

    // Build CSV content
    let csv = "URL,Verdict,Risk Score,Date/Time\n";
    history.forEach(item => {
      const time = new Date(item.time).toLocaleString();
      // wrap URL in quotes in case it contains commas
      csv += `"${item.url}","${item.prediction}",${item.score},"${time}"\n`;
    });

    // Trigger download
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "phishing_scan_history.csv";
    a.click();
    URL.revokeObjectURL(url);
  });
});

// ── Open dashboard in a new tab ───────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const dashBtn = document.getElementById("dashboard-btn");
  if (dashBtn) {
    dashBtn.addEventListener("click", () => {
      chrome.tabs.create({ url: "http://127.0.0.1:5000/dashboard" });
    });
  }
});

// ── Open reports page ─────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const rBtn = document.getElementById("reports-btn");
  if (rBtn) {
    rBtn.addEventListener("click", () => {
      chrome.tabs.create({ url: chrome.runtime.getURL("report.html") });
    });
  }
});