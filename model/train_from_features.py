import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ── Load both featurized datasets ─────────────────────────
print("Loading featurized datasets...")
kaggle = pd.read_csv('../data/featurized_dataset.csv')
tranco = pd.read_csv('../data/featurized_tranco.csv')
print(f"  Kaggle featurized: {kaggle.shape}")
print(f"  Tranco featurized: {tranco.shape}")

# ── Check columns match ───────────────────────────────────
if list(kaggle.columns) != list(tranco.columns):
    print("\n  WARNING: column mismatch! Aligning...")
    common = [c for c in kaggle.columns if c in tranco.columns]
    kaggle = kaggle[common]
    tranco = tranco[common]
    print(f"  Aligned to {len(common)} common columns.")

# ── Combine ───────────────────────────────────────────────
combined = pd.concat([kaggle, tranco], ignore_index=True)
combined = combined.drop_duplicates(subset=['url'], keep='first')
print(f"\n  Combined featurized: {combined.shape}")
print(f"  Label distribution:\n{combined['label'].value_counts().to_string()}")

# ── Separate features (X) and label (y) ───────────────────
# Drop non-feature columns
X = combined.drop(columns=['url', 'label'])
y = combined['label']
feature_names = list(X.columns)
print(f"\n  Number of numerical features: {len(feature_names)}")

# ── Train/test split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Hyperparameter tuning ─────────────────────────────────
print("\nRunning hyperparameter tuning...")
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=3,
                            scoring='f1', random_state=42, verbose=2)
search.fit(X_train, y_train)
print(f"\nBest params: {search.best_params_}")

# ── Evaluate ──────────────────────────────────────────────
best = search.best_estimator_
y_pred = best.predict(X_test)
print("\n── Evaluation ──")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("\n", classification_report(y_test, y_pred, target_names=['Safe','Phishing']))

# ── Save ──────────────────────────────────────────────────
joblib.dump(best, 'phishing_model.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
print("\nModel saved to phishing_model.pkl")