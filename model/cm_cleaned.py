import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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

model = joblib.load('phishing_model_cleaned_only.pkl')
feat_names = joblib.load('feature_names_cleaned_only.pkl')
X_test = X_test[feat_names]

print("Predicting...")
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (cleaned-only model):")
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['Safe','Phishing']))

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=['Safe','Phishing']).plot(
    ax=ax, cmap='Blues', values_format='d', colorbar=True)
plt.title('Confusion Matrix - Cleaned Dataset Model')
plt.tight_layout()
plt.savefig('../data/eda_charts/confusion_cleaned_only.png', dpi=120)
print("\nSaved: confusion_cleaned_only.png")
plt.show()