import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print("Loading featurized data and model...")
kaggle = pd.read_csv('../data/featurized_dataset.csv')
tranco = pd.read_csv('../data/featurized_tranco.csv')
data = pd.concat([kaggle, tranco], ignore_index=True).drop_duplicates(subset=['url'])

model = joblib.load('phishing_model.pkl')
feature_names = joblib.load('feature_names.pkl')

X = data[feature_names]
y = data['label']
urls = data['url'].values

X_train, X_test, y_train, y_test, url_train, url_test = train_test_split(
    X, y, urls, test_size=0.2, random_state=42, stratify=y
)

# ── 1. FEATURE IMPORTANCE ─────────────────────────────────
print("\n" + "="*55)
print("  FEATURE IMPORTANCE (what the model relies on)")
print("="*55)
imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print(imp.to_string())

print("\n  Features contributing almost nothing (<0.005):")
weak = imp[imp < 0.005]
print("   ", list(weak.index) if len(weak) else "None")

# ── 2. ERROR ANALYSIS ─────────────────────────────────────
print("\n" + "="*55)
print("  ERROR ANALYSIS (where the model goes wrong)")
print("="*55)
y_pred = model.predict(X_test)

# False Positives: predicted phishing (1), actually safe (0)
fp_mask = (y_pred == 1) & (y_test.values == 0)
# False Negatives: predicted safe (0), actually phishing (1)
fn_mask = (y_pred == 0) & (y_test.values == 1)

print(f"\n  False Positives: {fp_mask.sum()}  (safe sites flagged as phishing)")
print(f"  False Negatives: {fn_mask.sum()}  (phishing missed)")

# Compare feature averages: FP vs correct-safe, FN vs correct-phishing
X_test_reset = X_test.reset_index(drop=True)
correct_safe = (y_pred == 0) & (y_test.values == 0)
correct_phish = (y_pred == 1) & (y_test.values == 1)

print("\n  --- FALSE POSITIVES: how do wrongly-flagged safe sites differ? ---")
print("  (feature averages: false positives vs correctly-identified safe)")
fp_profile = pd.DataFrame({
    'false_positive_avg': X_test_reset[fp_mask].mean(),
    'correct_safe_avg': X_test_reset[correct_safe].mean()
})
fp_profile['difference'] = fp_profile['false_positive_avg'] - fp_profile['correct_safe_avg']
print(fp_profile.reindex(fp_profile['difference'].abs().sort_values(ascending=False).index).head(8).to_string())

print("\n  --- FALSE NEGATIVES: how do missed phishing differ? ---")
print("  (feature averages: false negatives vs correctly-caught phishing)")
fn_profile = pd.DataFrame({
    'false_negative_avg': X_test_reset[fn_mask].mean(),
    'correct_phishing_avg': X_test_reset[correct_phish].mean()
})
fn_profile['difference'] = fn_profile['false_negative_avg'] - fn_profile['correct_phishing_avg']
print(fn_profile.reindex(fn_profile['difference'].abs().sort_values(ascending=False).index).head(8).to_string())

# ── 3. Sample of misclassified URLs ───────────────────────
print("\n" + "="*55)
print("  SAMPLE MISCLASSIFIED URLs")
print("="*55)
print("\n  False Positives (safe URLs wrongly flagged):")
for u in url_test[fp_mask][:8]:
    print(f"    {str(u)[:75]}")
print("\n  False Negatives (phishing URLs missed):")
for u in url_test[fn_mask][:8]:
    print(f"    {str(u)[:75]}")

# ── 4. FEATURE CORRELATION (redundancy between columns) ───
print("\n" + "="*55)
print("  FEATURE CORRELATION (redundancy between columns)")
print("="*55)

corr = X.corr()

# Heatmap figure for the report
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=150)
print("\n  Heatmap saved to feature_correlation.png")

# The actual answer: strongly-correlated (redundant) pairs
threshold = 0.85
cols = corr.columns
pairs = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr.iloc[i, j]
        if abs(r) >= threshold:
            pairs.append((cols[i], cols[j], round(float(r), 3)))

print(f"\n  Feature pairs with |correlation| >= {threshold}:")
if pairs:
    for a, b, r in sorted(pairs, key=lambda p: -abs(p[2])):
        print(f"    {a:<28} {b:<28} r = {r}")
else:
    print("    None — no strongly redundant features.")