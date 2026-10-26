import joblib
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from features import extract_features

print("Loading data and model...")
df = pd.read_csv('../data/combined_dataset.csv')

def encode_label(label):
    if label == 'phishing': return 1
    elif label == 'malware': return 1
    else: return 0
df['label'] = df['type'].apply(encode_label)

print("Extracting features (a few minutes)...")
X = pd.DataFrame([extract_features(str(u)) for u in df['url']])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load('phishing_model.pkl')
feat_names = joblib.load('feature_names.pkl')
X_test = X_test[feat_names]

print("Computing metrics...")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

metrics = {
    'accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
    'precision': round(precision_score(y_test, y_pred) * 100, 2),
    'recall':    round(recall_score(y_test, y_pred) * 100, 2),
    'f1':        round(f1_score(y_test, y_pred) * 100, 2),
    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    'test_size': int(len(y_test))
}

# Save to JSON
with open('../data/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nMetrics saved to data/metrics.json:")
print(json.dumps(metrics, indent=2))