import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from features import extract_features

print("Loading CLEANED dataset (no Tranco)...")
df = pd.read_csv('../data/cleaned_dataset.csv')

def encode_label(label):
    if label == 'phishing': return 1
    elif label == 'malware': return 1
    else: return 0  # benign + defacement = safe
df['label'] = df['type'].apply(encode_label)

print("Extracting features (a few minutes)...")
X = pd.DataFrame([extract_features(str(u)) for u in df['url']])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest (cleaned-only)...")
model = RandomForestClassifier(
    n_estimators=100, max_depth=30, min_samples_split=5,
    random_state=42, n_jobs=-1, class_weight='balanced'
)
model.fit(X_train, y_train)

joblib.dump(model, 'phishing_model_cleaned_only.pkl')
joblib.dump(list(X.columns), 'feature_names_cleaned_only.pkl')
print("Saved: phishing_model_cleaned_only.pkl")