import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from features import extract_features

print("Loading data and model...")
df = pd.read_csv('../data/combined_dataset.csv')

def encode_label(label):
    if label == 'phishing':
        return 1
    elif label == 'malware':
        return 1  # malware treated as malicious
    else:
        return 0  # benign and defacement = safe
df['label'] = df['type'].apply(encode_label)

print("Extracting features (this takes a few minutes)...")
X = pd.DataFrame([extract_features(str(u)) for u in df['url']])
y = df['label']

# Same split as training (random_state=42) so we evaluate the held-out test portion
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load('phishing_model.pkl')
feature_names = joblib.load('feature_names.pkl')
X_test = X_test[feature_names]

print("Predicting...")
y_pred = model.predict(X_test)

# ── Build confusion matrix ────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (Safe correctly safe):     {tn}")
print(f"False Positives (Safe flagged phishing):  {fp}")
print(f"False Negatives (Phishing missed):        {fn}")
print(f"True Positives (Phishing caught):         {tp}")

# ── Plot it as a heatmap ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Safe', 'Phishing']
)
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
plt.title('Confusion Matrix - Phishing Detection Model')
plt.tight_layout()
plt.savefig('../data/eda_charts/confusion_matrix.png', dpi=120)
print("\nSaved: confusion_matrix.png")
plt.show()