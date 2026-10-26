import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, confusion_matrix,
                             precision_recall_curve)

print("Loading featurized data and model...")
kaggle = pd.read_csv('../data/featurized_dataset.csv')
tranco = pd.read_csv('../data/featurized_tranco.csv')
data = pd.concat([kaggle, tranco], ignore_index=True).drop_duplicates(subset=['url'])

model = joblib.load('phishing_model.pkl')
feature_names = joblib.load('feature_names.pkl')

X = data[feature_names]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Getting probability predictions...")
# Probability of the positive class (phishing)
y_proba = model.predict_proba(X_test)[:, 1]

# ── Evaluate across thresholds ────────────────────────────
print("\n" + "="*72)
print("  THRESHOLD ANALYSIS")
print("="*72)
print(f"{'Thresh':>7}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'FP':>8}{'FN':>8}")
print("-"*72)

results = []
for t in np.arange(0.05, 1.0, 0.05):
    y_pred = (y_proba >= t).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results.append({'threshold': round(t,2), 'accuracy': acc, 'precision': prec,
                    'recall': rec, 'f1': f1, 'fp': fp, 'fn': fn})
    print(f"{t:>7.2f}{acc:>10.4f}{prec:>11.4f}{rec:>9.4f}{f1:>8.4f}{fp:>8}{fn:>8}")

df = pd.DataFrame(results)

# ── Find optimal thresholds by different criteria ─────────
print("\n" + "="*72)
print("  OPTIMAL THRESHOLDS BY CRITERION")
print("="*72)

best_f1 = df.loc[df['f1'].idxmax()]
print(f"\n  Best F1 score:")
print(f"    Threshold: {best_f1['threshold']}  F1: {best_f1['f1']:.4f}")
print(f"    Precision: {best_f1['precision']:.4f}  Recall: {best_f1['recall']:.4f}")
print(f"    FP: {int(best_f1['fp'])}  FN: {int(best_f1['fn'])}")

best_acc = df.loc[df['accuracy'].idxmax()]
print(f"\n  Best accuracy:")
print(f"    Threshold: {best_acc['threshold']}  Accuracy: {best_acc['accuracy']:.4f}")
print(f"    FP: {int(best_acc['fp'])}  FN: {int(best_acc['fn'])}")

fewest_errors = df.loc[(df['fp'] + df['fn']).idxmin()]
print(f"\n  Fewest total errors (FP + FN):")
print(f"    Threshold: {fewest_errors['threshold']}")
print(f"    Total errors: {int(fewest_errors['fp'] + fewest_errors['fn'])}")
print(f"    FP: {int(fewest_errors['fp'])}  FN: {int(fewest_errors['fn'])}")

# Current default for comparison
current = df[df['threshold'] == 0.5].iloc[0]
print(f"\n  Current default (0.50):")
print(f"    Accuracy: {current['accuracy']:.4f}  F1: {current['f1']:.4f}")
print(f"    Precision: {current['precision']:.4f}  Recall: {current['recall']:.4f}")
print(f"    FP: {int(current['fp'])}  FN: {int(current['fn'])}")

# ── Plot 1: Precision-Recall curve ────────────────────────
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color='#2563eb', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../data/eda_charts/precision_recall_curve.png', dpi=120)
plt.close()
print("\n  Saved: precision_recall_curve.png")

# ── Plot 2: Metrics vs threshold ──────────────────────────
plt.figure(figsize=(9, 6))
plt.plot(df['threshold'], df['precision'], label='Precision', marker='o', color='#2563eb')
plt.plot(df['threshold'], df['recall'], label='Recall', marker='s', color='#ef4444')
plt.plot(df['threshold'], df['f1'], label='F1 Score', marker='^', color='#22c55e')
plt.axvline(0.5, color='#94a3b8', linestyle='--', label='Default (0.5)')
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.title('Performance Metrics vs Classification Threshold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../data/eda_charts/threshold_metrics.png', dpi=120)
plt.close()
print("  Saved: threshold_metrics.png")

# ── Plot 3: Error counts vs threshold ─────────────────────
plt.figure(figsize=(9, 6))
plt.plot(df['threshold'], df['fp'], label='False Positives', marker='o', color='#f59e0b')
plt.plot(df['threshold'], df['fn'], label='False Negatives', marker='s', color='#8b5cf6')
plt.axvline(0.5, color='#94a3b8', linestyle='--', label='Default (0.5)')
plt.xlabel('Classification Threshold')
plt.ylabel('Number of Errors')
plt.title('False Positives and False Negatives vs Threshold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../data/eda_charts/threshold_errors.png', dpi=120)
plt.close()
print("  Saved: threshold_errors.png")

df.to_csv('../data/threshold_analysis.csv', index=False)
print("  Saved: threshold_analysis.csv")