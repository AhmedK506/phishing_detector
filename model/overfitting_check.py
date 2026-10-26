import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

print("Loading data and model...")
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

# ── Overfitting check: train vs test performance ──────────
print("\n" + "="*55)
print("  OVERFITTING / UNDERFITTING CHECK")
print("="*55)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_f1 = f1_score(y_train, train_pred)
test_f1 = f1_score(y_test, test_pred)

print(f"\n  Training accuracy: {train_acc:.4f}")
print(f"  Test accuracy:     {test_acc:.4f}")
print(f"  Gap:               {train_acc - test_acc:.4f}")
print(f"\n  Training F1:       {train_f1:.4f}")
print(f"  Test F1:           {test_f1:.4f}")
print(f"  Gap:               {train_f1 - test_f1:.4f}")

gap = train_acc - test_acc
print("\n  Interpretation:")
if gap > 0.10:
    print("    Large gap - model shows signs of OVERFITTING.")
elif gap > 0.05:
    print("    Moderate gap - mild overfitting may be present.")
elif test_acc < 0.70:
    print("    Both scores low - model may be UNDERFITTING.")
else:
    print("    Small gap and strong test performance -")
    print("    model generalises well. No significant overfitting.")

# ── Cross-validation for stability ────────────────────────
from sklearn.model_selection import StratifiedKFold, cross_val_score

print("\n  Running 5-fold stratified cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

print(f"\n  Fold scores: {[round(s, 4) for s in cv_scores]}")
print(f"  Mean accuracy: {cv_scores.mean():.4f}")
print(f"  Std deviation: {cv_scores.std():.4f}")
print(f"\n  Result: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")