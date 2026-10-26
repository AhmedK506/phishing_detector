chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "showWarning") {
    showWarningBanner(msg.prob);
  }
});

function showWarningBanner(prob) {
  if (document.getElementById("phishing-detector-banner")) return;

  // Inject animation keyframes once
  if (!document.getElementById("pd-banner-style")) {
    const style = document.createElement("style");
    style.id = "pd-banner-style";
    style.textContent = `
      @keyframes pdSlideDown {
        from { transform: translateY(-100%); }
        to   { transform: translateY(0); }
      }
      #pd-leave:hover { background:#fff !important; color:#dc2626 !important; }
      #pd-dismiss:hover { background:rgba(255,255,255,0.35) !important; }
    `;
    document.head.appendChild(style);
  }

  const banner = document.createElement("div");
  banner.id = "phishing-detector-banner";
  banner.style.cssText = `
    position: fixed; top: 0; left: 0; width: 100%;
    background: #dc2626; color: #fff; z-index: 2147483647;
    padding: 12px 20px; font-family: 'Segoe UI', sans-serif;
    font-size: 14px; display: flex; align-items: center; gap: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.35);
    animation: pdSlideDown 0.4s ease-out;
    box-sizing: border-box;
  `;

  banner.innerHTML = `
    <span style="display:flex; align-items:center; justify-content:center;
                 width:32px; height:32px; background:rgba(255,255,255,0.18);
                 border-radius:8px; flex-shrink:0;">
      <svg viewBox="0 0 24 24" width="20" height="20" fill="none"
           stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
      </svg>
    </span>
    <span style="flex:1; line-height:1.4;">
      <strong style="display:block; font-size:14px;">Warning: This site may be a phishing attempt</strong>
      <span style="font-size:12px; opacity:0.9;">Risk level ${Math.round(prob)}%. Do not enter passwords or personal information.</span>
    </span>
    <button id="pd-leave" style="
      background:rgba(255,255,255,0.95); border:none; color:#dc2626;
      padding:7px 14px; border-radius:6px; cursor:pointer; font-size:13px;
      font-weight:600; flex-shrink:0; transition:all 0.15s;">
      Leave this site
    </button>
    <button id="pd-dismiss" style="
      background:rgba(255,255,255,0.2); border:none; color:#fff;
      padding:7px 12px; border-radius:6px; cursor:pointer; font-size:13px;
      flex-shrink:0; transition:all 0.15s;">
      Dismiss
    </button>
  `;

  document.body.appendChild(banner);

  // Dismiss: slide back up and remove
  document.getElementById("pd-dismiss").addEventListener("click", () => {
    banner.style.transition = "transform 0.3s ease-in";
    banner.style.transform = "translateY(-100%)";
    setTimeout(() => banner.remove(), 300);
  });

  // Leave: go back to the previous page
  document.getElementById("pd-leave").addEventListener("click", () => {
    if (window.history.length > 1) {
      window.history.back();
    } else {
      window.location.href = "https://www.google.com";
    }
  });
}