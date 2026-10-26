import pandas as pd
import joblib
import sys
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

sys.path.append(os.path.dirname(__file__))
from features import extract_features

# ── Config ────────────────────────────────────────────────
N_PER_CLASS = 50   # 50 legit + 50 phishing = 100 test URLs
TRANCO_PATH = '../data/tranco_top.csv'
PHISHTANK_PATH = '../data/verified_online.csv'

# ── 1. Build held-out LEGIT set (Tranco ranks 30k-60k) ────
# These were NOT used in training (we trained on top 30k only)
print("Building held-out legitimate test set...")
tranco = pd.read_csv(TRANCO_PATH, header=None, names=['rank', 'domain'])
heldout_legit = tranco.iloc[30000:60000]              # ranks 30,001-60,000
legit_sample = heldout_legit.sample(n=N_PER_CLASS, random_state=42)
legit_urls = [f"https://www.{d}" for d in legit_sample['domain']]
print(f"  {len(legit_urls)} legit URLs (never seen in training)")

# ── 2. Build PHISHING set (PhishTank verified) ────────────
print("Building phishing test set...")
phish = pd.read_csv(PHISHTANK_PATH)
phish_sample = phish.sample(n=N_PER_CLASS, random_state=42)
phish_urls = phish_sample['url'].tolist()
print(f"  {len(phish_urls)} verified phishing URLs from PhishTank")

# ── 3. Assemble labelled test set ─────────────────────────
# label: 0 = legit/safe, 1 = phishing
test_urls = legit_urls + phish_urls
test_labels = [0] * len(legit_urls) + [1] * len(phish_urls)

# ── 4. Extract features for all test URLs ─────────────────
print("\nExtracting features for test URLs...")
X_test = pd.DataFrame([extract_features(str(u)) for u in test_urls])
y_test = test_labels

# ── 5. Helper: evaluate one model ─────────────────────────
def evaluate_model(model_path, feature_names_path, label):
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    X = X_test[feature_names]    # align column order

    # Base model predictions
    raw_pred = model.predict(X)

    # Apply the same path-impersonation heuristic the API uses
    y_pred = []
    for i in range(len(X)):
        row = X.iloc[i]
        brand_in_path = row.get('brand_in_path', 0)
        suspicious_kw = row.get('suspicious_path_keywords', 0)
        embedded_dom = row.get('embedded_domain_in_path', 0)

        if brand_in_path >= 1 and suspicious_kw >= 2 and embedded_dom >= 1:
            y_pred.append(1)            # heuristic forces phishing
        else:
            y_pred.append(raw_pred[i])  # otherwise trust the model
    y_pred = pd.Series(y_pred)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"\n  False Positives (legit flagged as phishing): {fp}")
    print(f"  False Negatives (phishing missed):           {fn}")
    print(f"  True Positives (phishing caught):            {tp}")
    print(f"  True Negatives (legit correctly safe):       {tn}")

    return {
        'label': label, 'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'fp': fp, 'fn': fn
    }

# ── 6. Run both models ────────────────────────────────────
print("\n" + "#"*50)
print("# BEFORE vs AFTER MODEL COMPARISON")
print("#"*50)

old = evaluate_model(
    '../model/phishing_model_OLD.pkl',
    '../model/feature_names_OLD.pkl',
    'Model test'
)

new = evaluate_model(
    '../model/phishing_model.pkl',
    '../model/feature_names.pkl',
    'Model Test2'
)

# ── 7. Summary table ──────────────────────────────────────
print("\n" + "="*50)
print("  SUMMARY")
print("="*50)
print(f"{'Metric':<18}{'OLD':>10}{'NEW':>10}{'Change':>12}")
print("-"*50)
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    o, n = old[metric], new[metric]
    change = n - o
    print(f"{metric.capitalize():<18}{o:>10.4f}{n:>10.4f}{change:>+12.4f}")
print(f"{'False Positives':<18}{old['fp']:>10}{new['fp']:>10}{new['fp']-old['fp']:>+12}")
print(f"{'False Negatives':<18}{old['fn']:>10}{new['fn']:>10}{new['fn']-old['fn']:>+12}")