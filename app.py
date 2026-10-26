from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import joblib
import pandas as pd
import sys
import os
import tldextract
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from features import extract_features
from html_analyzer import analyze_html
from visual_analyzer import analyze_visual

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Phishing Detector — Model Dashboard</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; font-family:'Segoe UI',system-ui,sans-serif; }
  body { background:#0f172a; color:#e2e8f0; padding:32px; }
  .wrap { max-width:1100px; margin:0 auto; }
  .head h1 { font-size:26px; color:#fff; display:flex; align-items:center; gap:10px; }
  .head .sub { color:#94a3b8; font-size:14px; margin-top:4px; }
  .tag { display:inline-block; background:#1e293b; color:#93c5fd; font-size:11px;
         padding:3px 10px; border-radius:20px; margin-right:6px; margin-top:8px; }

  /* Loading overlay */
  #loading { text-align:center; padding:80px 20px; }
  .spinner { width:52px; height:52px; margin:0 auto 20px;
    border:5px solid #1e293b; border-top:5px solid #3b82f6;
    border-radius:50%; animation:spin 0.9s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  #loading .msg { font-size:16px; color:#cbd5e1; font-weight:500; }
  .dots::after { content:''; animation:dots 1.4s steps(4,end) infinite; }
  @keyframes dots {
    0% { content:''; } 25% { content:'.'; }
    50% { content:'..'; } 75% { content:'...'; } 100% { content:''; }
  }
  #loading .hint { font-size:12px; color:#64748b; margin-top:12px; }

  #content { display:none; }
  .grid4 { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:24px; }
  .metric { background:linear-gradient(135deg,#1e293b,#172033); border:1px solid #2d3b52;
            border-radius:14px; padding:18px; }
  .metric .label { font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:#94a3b8; }
  .metric .val { font-size:32px; font-weight:700; color:#fff; margin-top:6px; }
  .metric .val span { font-size:16px; color:#64748b; }
  .metric.accent .val { color:#4ade80; }

  .row2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px; }
  .panel { background:#1e293b; border:1px solid #2d3b52; border-radius:14px; padding:20px; }
  .panel h2 { font-size:14px; color:#fff; margin-bottom:16px; display:flex; align-items:center; gap:8px; }
  .panel h2 .dot { width:8px; height:8px; border-radius:50%; background:#3b82f6; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  td { padding:8px 2px; border-bottom:1px solid #2a3854; color:#cbd5e1; }
  td:last-child { text-align:right; font-weight:600; color:#fff; }
  .ok { background:#14352a; color:#4ade80; padding:2px 8px; border-radius:5px; font-size:12px; }

  .cm { display:grid; grid-template-columns:auto 1fr 1fr; gap:6px; align-items:center; }
  .cm .lbl { font-size:11px; color:#94a3b8; text-align:center; }
  .cm .cell { border-radius:8px; padding:16px 8px; text-align:center; }
  .cm .cell .n { font-size:22px; font-weight:700; color:#fff; }
  .cm .cell .t { font-size:10px; color:rgba(255,255,255,.7); margin-top:2px; }
  .cell.correct { background:#1d4ed8; }
  .cell.error { background:#334155; }

  .cmp { margin-bottom:16px; }
  .cmp .cl { display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px; color:#cbd5e1; }
  .cmp .track { display:flex; flex-direction:column; gap:4px; }
  .cmp .barrow { display:flex; align-items:center; gap:8px; }
  .cmp .tag2 { font-size:10px; width:42px; color:#94a3b8; }
  .cmp .barwrap { flex:1; height:8px; background:#0f172a; border-radius:5px; overflow:hidden; }
  .cmp .old { height:100%; background:#64748b; }
  .cmp .new { height:100%; background:#4ade80; }
  .highlight { font-size:13px; color:#4ade80; font-weight:600; }
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <h1>🛡️ Phishing Detector — Model Dashboard</h1>
    <div class="sub">Machine Learning Analysis & Performance Overview</div>
  </div>

  <!-- Loading state -->
  <div id="loading">
    <div class="spinner"></div>
    <div class="msg">Running analysis<span class="dots"></span></div>
    <div class="hint">Evaluating the model on the full dataset — this may take a few minutes.</div>
  </div>

  <!-- Dashboard content (hidden until data loads) -->
  <div id="content"></div>
</div>

<script>
fetch('/dashboard/data')
  .then(r => r.json())
  .then(d => {
    document.getElementById('loading').style.display = 'none';
    const c = document.getElementById('content');
    c.style.display = 'block';
    c.innerHTML = `
      <div style="margin-bottom:20px;">
        <span class="tag">${d.model.algorithm}</span>
        <span class="tag">${d.model.type}</span>
        <span class="tag">${d.model.tuning}</span>
        <span class="tag">${d.model.balancing}</span>
      </div>
      <div class="grid4">
        <div class="metric accent"><div class="label">Accuracy</div><div class="val">${d.metrics.accuracy}<span>%</span></div></div>
        <div class="metric"><div class="label">Precision</div><div class="val">${d.metrics.precision}<span>%</span></div></div>
        <div class="metric"><div class="label">Recall</div><div class="val">${d.metrics.recall}<span>%</span></div></div>
        <div class="metric"><div class="label">F1 Score</div><div class="val">${d.metrics.f1}<span>%</span></div></div>
      </div>
      <div class="row2">
        <div class="panel">
          <h2><span class="dot"></span>Data Preprocessing</h2>
          <table>
            <tr><td>Raw dataset rows</td><td>${d.preprocessing.raw_rows.toLocaleString()}</td></tr>
            <tr><td>Malformed URLs removed</td><td>${d.preprocessing.malformed_removed}</td></tr>
            <tr><td>Duplicate URLs removed</td><td>${d.preprocessing.duplicates_removed.toLocaleString()}</td></tr>
            <tr><td>Missing values</td><td><span class="ok">None</span></td></tr>
            <tr><td>Cleaned rows</td><td>${d.preprocessing.cleaned_rows.toLocaleString()}</td></tr>
          </table>
        </div>
        <div class="panel">
          <h2><span class="dot"></span>Dataset Composition</h2>
          <table>
            <tr><td>Kaggle (cleaned)</td><td>${d.dataset.kaggle_rows.toLocaleString()}</td></tr>
            <tr><td>Tranco legit added</td><td>${d.dataset.tranco_added.toLocaleString()}</td></tr>
            <tr><td>Total training rows</td><td>${d.dataset.total_training.toLocaleString()}</td></tr>
            <tr><td>Class balance</td><td>${d.dataset.imbalance}</td></tr>
            <tr><td>Numerical features</td><td>${d.dataset.num_features}</td></tr>
          </table>
        </div>
      </div>
      <div class="row2">
        <div class="panel">
          <h2><span class="dot"></span>Confusion Matrix</h2>
          <div class="cm">
            <div></div><div class="lbl">Pred. Safe</div><div class="lbl">Pred. Phishing</div>
            <div class="lbl">Actual<br>Safe</div>
            <div class="cell correct"><div class="n">${d.confusion.tn.toLocaleString()}</div><div class="t">True Negative</div></div>
            <div class="cell error"><div class="n">${d.confusion.fp.toLocaleString()}</div><div class="t">False Positive</div></div>
            <div class="lbl">Actual<br>Phishing</div>
            <div class="cell error"><div class="n">${d.confusion.fn.toLocaleString()}</div><div class="t">False Negative</div></div>
            <div class="cell correct"><div class="n">${d.confusion.tp.toLocaleString()}</div><div class="t">True Positive</div></div>
          </div>
        </div>
        <div class="panel">
          <h2><span class="dot"></span>Improvement (Held-out Test)</h2>
          <div class="cmp">
            <div class="cl"><span>Accuracy</span><span>${d.evaluation.old_accuracy}% → ${d.evaluation.new_accuracy}%</span></div>
            <div class="track">
              <div class="barrow"><span class="tag2">Before</span><div class="barwrap"><div class="old" style="width:${d.evaluation.old_accuracy}%"></div></div></div>
              <div class="barrow"><span class="tag2">After</span><div class="barwrap"><div class="new" style="width:${d.evaluation.new_accuracy}%"></div></div></div>
            </div>
          </div>
          <div class="cmp">
            <div class="cl"><span>Recall</span><span>${d.evaluation.old_recall}% → ${d.evaluation.new_recall}%</span></div>
            <div class="track">
              <div class="barrow"><span class="tag2">Before</span><div class="barwrap"><div class="old" style="width:${d.evaluation.old_recall}%"></div></div></div>
              <div class="barrow"><span class="tag2">After</span><div class="barwrap"><div class="new" style="width:${d.evaluation.new_recall}%"></div></div></div>
            </div>
          </div>
          <div style="margin-top:14px;"><div class="highlight">False positives: ${d.evaluation.old_fp} → ${d.evaluation.new_fp} on 50 legit sites</div></div>
        </div>
      </div>
    `;
  })
  .catch(err => {
    document.getElementById('loading').innerHTML =
      '<div class="msg" style="color:#ef4444;">Failed to load. Is the server running?</div>';
  });
</script>
</body>
</html>
"""

app = Flask(__name__)
CORS(app)

model = joblib.load('model/phishing_model.pkl')
feature_names = joblib.load('model/feature_names.pkl')

# ── Whitelist ─────────────────────────────────────────────
WHITELIST = [
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
    'instagram.com', 'linkedin.com', 'microsoft.com', 'apple.com',
    'amazon.com', 'wikipedia.org', 'github.com', 'stackoverflow.com',
    'reddit.com', 'netflix.com', 'spotify.com', 'dropbox.com',
    'yahoo.com', 'bing.com', 'office.com', 'live.com',
    'outlook.com', 'gmail.com', 'whatsapp.com', 'tiktok.com',
    # ── Regional / SEA sites ──
    'shopee.com.my', 'shopee.sg', 'lazada.com.my', 'lazada.sg',
    'maybank2u.com.my', 'cimbclicks.com.my', 'publicbank.com.my',
    'apu.edu.my', 'grab.com', 'foodpanda.com.my',
    'astro.com.my', 'tnb.com.my', 'pos.com.my'
]

def is_whitelisted(url):
    ext = tldextract.extract(url)
    full_domain = f"{ext.domain}.{ext.suffix}"
    return full_domain in WHITELIST

# ── Path-impersonation heuristic (strict: 3 signals required) ──
def path_impersonation_override(features):
    """
    Conservative hybrid rule. Fires ONLY when multiple independent
    path-impersonation signals co-occur — a pattern legitimate sites
    essentially never exhibit together. Returns (triggered, reason).
    """
    brand_in_path = features.get('brand_in_path', 0)
    suspicious_kw = features.get('suspicious_path_keywords', 0)
    embedded_dom = features.get('embedded_domain_in_path', 0)

    # Strict: require a brand name in the path AND phishing-kit
    # keywords AND an embedded fake domain, all at once.
    if brand_in_path >= 1 and suspicious_kw >= 2 and embedded_dom >= 1:
        reason = (
            f'Path-impersonation pattern detected: brand name in path, '
            f'{suspicious_kw} phishing keywords, and an embedded domain '
            f'in the URL path'
        )
        return True, reason

    return False, None

# ── Ensemble scoring ──────────────────────────────────────
def ensemble_decision(url_score, html_result, visual_result):
    # Weights for each component
    URL_WEIGHT = 0.5
    HTML_WEIGHT = 0.3
    VISUAL_WEIGHT = 0.2

    url_phishing_prob = url_score

    html_score = html_result.get('risk_score', 0) if html_result.get('success') else url_score
    visual_score = visual_result.get('risk_score', 0) if visual_result.get('success') else 0

    # Weighted ensemble score
    final_score = (
        url_phishing_prob * URL_WEIGHT +
        html_score * HTML_WEIGHT +
        visual_score * VISUAL_WEIGHT
    )

    if final_score >= 60:
        risk_level = 'High'
        prediction = 'Phishing'
    elif final_score >= 35:
        risk_level = 'Medium'
        prediction = 'Suspicious'
    else:
        risk_level = 'Low'
        prediction = 'Safe'

    return round(final_score, 2), risk_level, prediction

# ── Routes ────────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({'status': 'Phishing Detector API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    url = data['url']
    mode = data.get('mode', 'full')  # 'full' or 'url_only'

    # Whitelist check
    if is_whitelisted(url):
        return jsonify({
            'url': url,
            'prediction': 'Safe',
            'risk_level': 'Low',
            'final_score': 0.0,
            'note': 'Whitelisted domain',
            'components': {
                'url_analysis': {'phishing_probability': 0.0},
                'html_analysis': {'risk_score': 0, 'indicators': []},
                'visual_analysis': {'risk_score': 0, 'indicators': []}
            }
        })

    # ── Component 1: URL Analysis ─────────────────────────
    features = extract_features(url)
    features_df = pd.DataFrame([features])[feature_names]
    prediction_ml = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]
    url_phishing_prob = round(float(probability[1]) * 100, 2)

    # ── Hybrid heuristic layer: path-impersonation override ──
    impersonation_triggered, impersonation_reason = path_impersonation_override(features)
    if impersonation_triggered:
        # Elevate the URL score — but only as high as needed to flag,
        # not slammed to 100, so it stays honest about being a heuristic
        url_phishing_prob = max(url_phishing_prob, 85.0)

    if mode == 'url_only':
        return jsonify({
            'url': url,
            'prediction': 'Phishing' if url_phishing_prob >= 50 else 'Safe',
            'confidence': round(float(max(probability)) * 100, 2),
            'phishing_probability': url_phishing_prob,
            'heuristic_triggered': impersonation_triggered,
            'heuristic_reason': impersonation_reason
        })

    # ── Component 2: HTML Analysis ────────────────────────
    html_result = analyze_html(url)

    # ── Component 3: Visual Analysis ─────────────────────
    visual_result = analyze_visual(url)

    # ── Ensemble Decision ─────────────────────────────────
    final_score, risk_level, final_prediction = ensemble_decision(
        url_phishing_prob, html_result, visual_result
    )

    return jsonify({
        'url': url,
        'prediction': final_prediction,
        'risk_level': risk_level,
        'final_score': final_score,
        'components': {
            'url_analysis': {
                'phishing_probability': url_phishing_prob,
                'prediction': 'Phishing' if prediction_ml == 1 else 'Safe'
            },
            'html_analysis': {
                'risk_score': html_result.get('risk_score', 0),
                'risk_level': html_result.get('risk_level', 'Unknown'),
                'indicators': html_result.get('indicators', []),
                'success': html_result.get('success', False)
            },
            'visual_analysis': {
                'risk_score': visual_result.get('risk_score', 0),
                'risk_level': visual_result.get('risk_level', 'Unknown'),
                'indicators': visual_result.get('indicators', []),
                'closest_brand': visual_result.get('closest_brand'),
                'similarity_score': visual_result.get('similarity_score', 0),
                'success': visual_result.get('success', False)
            }
        }
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# The page — loads instantly, shows animation
@app.route('/dashboard')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

# The data endpoint — does the slow computation, returns JSON
@app.route('/dashboard/data')
def dashboard_data():
    kaggle = pd.read_csv('data/featurized_dataset.csv')
    tranco = pd.read_csv('data/featurized_tranco.csv')
    data = pd.concat([kaggle, tranco], ignore_index=True).drop_duplicates(subset=['url'])

    X = data[feature_names]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return jsonify({
        "preprocessing": {"raw_rows": 651191, "malformed_removed": 61,
                          "duplicates_removed": 10071, "cleaned_rows": 641059},
        "dataset": {"kaggle_rows": 641059, "tranco_added": 30000,
                    "total_training": int(len(X)), "imbalance": "2.01 : 1",
                    "num_features": int(len(feature_names))},
        "metrics": {
            "accuracy": round(accuracy_score(y_test, y_pred)*100, 2),
            "precision": round(precision_score(y_test, y_pred)*100, 2),
            "recall": round(recall_score(y_test, y_pred)*100, 2),
            "f1": round(f1_score(y_test, y_pred)*100, 2)},
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "evaluation": {"old_accuracy": 47.0, "new_accuracy": 99.0,
                       "old_fp": 50, "new_fp": 0, "old_recall": 94.0, "new_recall": 98.0},
        "model": {"algorithm": "Random Forest",
                  "type": "Supervised · Binary Classification",
                  "tuning": "RandomizedSearchCV",
                  "balancing": "class_weight='balanced'"}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)