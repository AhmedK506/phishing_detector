"""
Standalone demo dashboard for the phishing detection model.

Run from the project root:   python demo_dashboard.py
Then open:                   http://127.0.0.1:5001

Computes metrics live from phishing_model.pkl and the featurized CSVs.
Independent of app.py — nothing here touches your main application.
"""

from flask import Flask, render_template_string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Model Performance</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; font-family:'Segoe UI',system-ui,sans-serif; }
  body { background:#0f172a; color:#e2e8f0; padding:40px; }
  .wrap { max-width:1000px; margin:0 auto; }
  h1 { font-size:26px; color:#fff; margin-bottom:4px; }
  .sub { color:#94a3b8; font-size:14px; margin-bottom:28px; }
  .grid4 { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:28px; }
  .metric { background:linear-gradient(135deg,#1e293b,#172033); border:1px solid #2d3b52;
            border-radius:14px; padding:22px; }
  .metric .label { font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:#94a3b8; }
  .metric .val { font-size:36px; font-weight:700; color:#fff; margin-top:8px; }
  .metric .val span { font-size:18px; color:#64748b; }
  .metric.accent .val { color:#4ade80; }
  .row2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  .panel { background:#1e293b; border:1px solid #2d3b52; border-radius:14px; padding:24px; }
  .panel h2 { font-size:15px; color:#fff; margin-bottom:18px; display:flex; align-items:center; gap:8px; }
  .panel h2 .dot { width:8px; height:8px; border-radius:50%; background:#3b82f6; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  td { padding:10px 2px; border-bottom:1px solid #2a3854; color:#cbd5e1; }
  td:last-child { text-align:right; font-weight:600; color:#fff; }
  .cm { display:grid; grid-template-columns:auto 1fr 1fr; gap:8px; align-items:center; }
  .cm .lbl { font-size:11px; color:#94a3b8; text-align:center; }
  .cm .cell { border-radius:8px; padding:20px 8px; text-align:center; }
  .cm .cell .n { font-size:26px; font-weight:700; color:#fff; }
  .cm .cell .t { font-size:10px; color:rgba(255,255,255,.7); margin-top:3px; }
  .cell.correct { background:#1d4ed8; }
  .cell.error { background:#334155; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🛡️ Model Performance Dashboard</h1>
  <div class="sub">Random Forest classifier · evaluated live on the held-out test set</div>

  <div class="grid4">
    <div class="metric accent"><div class="label">Accuracy</div><div class="val">{{ m.accuracy }}<span>%</span></div></div>
    <div class="metric"><div class="label">Precision</div><div class="val">{{ m.precision }}<span>%</span></div></div>
    <div class="metric"><div class="label">Recall</div><div class="val">{{ m.recall }}<span>%</span></div></div>
    <div class="metric"><div class="label">F1 Score</div><div class="val">{{ m.f1 }}<span>%</span></div></div>
  </div>

  <div class="row2">
    <div class="panel">
      <h2><span class="dot"></span>Confusion Matrix</h2>
      <div class="cm">
        <div></div><div class="lbl">Pred. Safe</div><div class="lbl">Pred. Phishing</div>
        <div class="lbl">Actual<br>Safe</div>
        <div class="cell correct"><div class="n">{{ c.tn }}</div><div class="t">True Negative</div></div>
        <div class="cell error"><div class="n">{{ c.fp }}</div><div class="t">False Positive</div></div>
        <div class="lbl">Actual<br>Phishing</div>
        <div class="cell error"><div class="n">{{ c.fn }}</div><div class="t">False Negative</div></div>
        <div class="cell correct"><div class="n">{{ c.tp }}</div><div class="t">True Positive</div></div>
      </div>
    </div>
    <div class="panel">
      <h2><span class="dot"></span>Dataset</h2>
      <table>
        <tr><td>Total samples</td><td>{{ d.total }}</td></tr>
        <tr><td>Test set size</td><td>{{ d.test }}</td></tr>
        <tr><td>Numerical features</td><td>{{ d.features }}</td></tr>
        <tr><td>Class balance (safe : phishing)</td><td>{{ d.balance }}</td></tr>
        <tr><td>Model</td><td>Random Forest</td></tr>
      </table>
    </div>
  </div>
</div>
</body>
</html>
"""


def compute():
    model = joblib.load('model/phishing_model.pkl')
    feature_names = joblib.load('model/feature_names.pkl')

    kaggle = pd.read_csv('data/featurized_dataset.csv')
    tranco = pd.read_csv('data/featurized_tranco.csv')
    data = pd.concat([kaggle, tranco], ignore_index=True).drop_duplicates(subset=['url'])

    X = data[feature_names]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    safe = int((y == 0).sum())
    phish = int((y == 1).sum())
    ratio = round(safe / phish, 2)

    m = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall": round(recall_score(y_test, y_pred) * 100, 2),
        "f1": round(f1_score(y_test, y_pred) * 100, 2),
    }
    c = {"tn": f"{tn:,}", "fp": f"{fp:,}", "fn": f"{fn:,}", "tp": f"{tp:,}"}
    d = {
        "total": f"{len(X):,}",
        "test": f"{len(X_test):,}",
        "features": len(feature_names),
        "balance": f"{ratio} : 1",
    }
    return m, c, d


@app.route('/')
def index():
    m, c, d = compute()
    return render_template_string(PAGE, m=m, c=c, d=d)


if __name__ == '__main__':
    print("Demo dashboard running at http://127.0.0.1:5001")
    app.run(debug=True, port=5001)