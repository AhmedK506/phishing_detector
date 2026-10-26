import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from features import extract_features

# ── 1. Load Dataset ───────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('../data/combined_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(df['type'].value_counts())

# ── 2. Label Encoding ─────────────────────────────────────
# We simplify to binary: phishing = 1, safe = 0
def encode_label(label):
    if label == 'phishing':
        return 1
    elif label == 'malware':
        return 1  # treat malware as malicious too
    else:
        return 0  # benign and defacement = safe

df['label'] = df['type'].apply(encode_label)

# ── 3. Feature Extraction ─────────────────────────────────
print("\nExtracting features from URLs (this may take a few minutes)...")

feature_list = []
for i, url in enumerate(df['url']):
    if i % 10000 == 0:
        print(f"  Processing URL {i}/{len(df)}...")
    feature_list.append(extract_features(str(url)))

X = pd.DataFrame(feature_list)
y = df['label']

print(f"\nFeature matrix shape: {X.shape}")

# ── 4. Train/Test Split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# ── 5. Hyperparameter Tuning ──────────────────────────────
print("\nRunning hyperparameter tuning...")
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
search = RandomizedSearchCV(
    rf, param_dist,
    n_iter=10,
    cv=3,
    scoring='f1',
    random_state=42,
    verbose=2
)
search.fit(X_train, y_train)
print(f"\nBest parameters: {search.best_params_}")

# ── 6. Evaluate Model ─────────────────────────────────────
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n── Model Evaluation ──────────────────────────")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))

# ── 7. Save Model ─────────────────────────────────────────
os.makedirs('../model', exist_ok=True)
joblib.dump(best_model, '../model/phishing_model.pkl')
joblib.dump(list(X.columns), '../model/feature_names.pkl')
print("\nModel saved to model/phishing_model.pkl")