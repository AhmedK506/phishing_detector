function fmtTime(ts) {
  return new Date(ts).toLocaleString();
}

function render() {
  chrome.storage.local.get({ scanHistory: [] }, (stored) => {
    const history = stored.scanHistory;
    const phishing = history.filter(h => h.prediction === "Phishing");

    document.getElementById("summary").innerHTML = `
      <div class="stat"><div class="n">${history.length}</div><div class="l">Total Scans</div></div>
      <div class="stat danger"><div class="n">${phishing.length}</div><div class="l">Phishing Detected</div></div>
      <div class="stat safe"><div class="n">${history.filter(h=>h.prediction==="Safe").length}</div><div class="l">Safe Sites</div></div>
    `;

    const hDiv = document.getElementById("history");
    if (history.length === 0) {
      hDiv.innerHTML = '<div class="empty">No scans recorded yet.</div>';
    } else {
      let rows = history.map(h => `
        <tr>
          <td class="url-cell" title="${h.url}">${h.url}</td>
          <td><span class="badge ${(h.prediction||'').toLowerCase()}">${h.prediction}</span></td>
          <td>${h.score}</td>
          <td>${fmtTime(h.time)}</td>
        </tr>`).join("");
      hDiv.innerHTML = `<table>
        <tr><th>URL</th><th>Verdict</th><th>Risk</th><th>Time</th></tr>
        ${rows}</table>`;
    }

    const rDiv = document.getElementById("reports");
    if (phishing.length === 0) {
      rDiv.innerHTML = '<div class="empty">No phishing threats detected. Nothing to report.</div>';
    } else {
      rDiv.innerHTML = phishing.map(h => `
        <div class="report">
          <div class="report-head">
            <div class="title">⚠ PHISHING THREAT REPORT</div>
            <div class="id">${h.id}</div>
          </div>
          <div class="field"><div class="k">Report ID</div><div class="v">${h.id}</div></div>
          <div class="field"><div class="k">Date / Time</div><div class="v">${fmtTime(h.time)}</div></div>
          <div class="field"><div class="k">Target URL</div><div class="v">${h.url}</div></div>
          <div class="field"><div class="k">Risk Level</div><div class="v"><span class="sev ${h.riskLevel}">${h.riskLevel}</span></div></div>
          <div class="field"><div class="k">Overall Risk Score</div><div class="v">${h.score} / 100</div></div>
          <div class="comps">
            <div class="comp"><div class="cn">${Math.round(h.urlScore)}%</div><div class="cl">URL Analysis</div></div>
            <div class="comp"><div class="cn">${h.htmlScore}</div><div class="cl">HTML Analysis</div></div>
            <div class="comp"><div class="cn">${h.visualScore}</div><div class="cl">Visual Analysis</div></div>
          </div>
          <div class="indicators">
            <div class="ih">Detection Indicators</div>
            ${(h.indicators && h.indicators.length) ? h.indicators.map(i => `<div class="ind">${i}</div>`).join("") : '<div class="ind">URL-based detection (no page-level indicators)</div>'}
          </div>
          <div class="rec">
            <strong>Recommendation:</strong> Do not enter credentials or personal information on this site.
            Avoid interacting with the page and report the URL to relevant authorities if received via email or message.
          </div>
        </div>
      `).join("");
    }
  });
}

document.getElementById("clear").addEventListener("click", () => {
  if (confirm("Clear all scan history and reports?")) {
    chrome.storage.local.set({ scanHistory: [] }, render);
  }
});

render();