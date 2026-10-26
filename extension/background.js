const API_URL = "http://127.0.0.1:5000/predict";

// Fast URL-only scan when a tab finishes loading
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url) {
    autoScan(tabId, tab.url);
  }
});

// Also scan when switching to an existing tab
chrome.tabs.onActivated.addListener((activeInfo) => {
  chrome.tabs.get(activeInfo.tabId, (tab) => {
    if (tab.url) autoScan(activeInfo.tabId, tab.url);
  });
});

async function autoScan(tabId, url) {
  // Check if auto-scan is enabled
  const settings = await chrome.storage.local.get({ autoScanEnabled: true });
  if (!settings.autoScanEnabled) {
    chrome.action.setBadgeText({ tabId, text: "" });
    return;  // auto-scan is off, do nothing
  }

  // Skip browser internal pages
  if (!url.startsWith("http")) {
    chrome.action.setBadgeText({ tabId, text: "" });
    return;
  }

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url, mode: "url_only" })  // FAST check
    });
    if (!res.ok) throw new Error();
    const data = await res.json();

    const prob = data.phishing_probability ?? 0;

    // Set badge colour based on result
    if (prob >= 60) {
      chrome.action.setBadgeText({ tabId, text: "!" });
      chrome.action.setBadgeBackgroundColor({ tabId, color: "#ef4444" }); // red
      // Tell the content script to show a warning banner
      chrome.tabs.sendMessage(tabId, { action: "showWarning", prob: prob });
    } else if (prob >= 35) {
      chrome.action.setBadgeText({ tabId, text: "?" });
      chrome.action.setBadgeBackgroundColor({ tabId, color: "#f59e0b" }); // amber
    } else {
      chrome.action.setBadgeText({ tabId, text: "✓" });
      chrome.action.setBadgeBackgroundColor({ tabId, color: "#22c55e" }); // green
    }
  } catch (err) {
    // Backend not running — clear badge
    chrome.action.setBadgeText({ tabId, text: "" });
  }
}