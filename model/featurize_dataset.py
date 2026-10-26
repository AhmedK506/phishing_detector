import pandas as pd
from features import extract_features

print("Loading cleaned dataset...")
df = pd.read_csv('../data/cleaned_dataset.csv')

def encode_label(label):
    if label == 'phishing': return 1
    elif label == 'malware': return 1
    else: return 0
df['label'] = df['type'].apply(encode_label)

print(f"Extracting numerical features from {len(df)} URLs (a few minutes)...")
features_list = []
for i, url in enumerate(df['url']):
    if i % 50000 == 0:
        print(f"  {i}/{len(df)}...")
    features_list.append(extract_features(str(url)))

# Build the numerical feature dataframe
feature_df = pd.DataFrame(features_list)

# Add back the label (and keep url for reference)
feature_df.insert(0, 'url', df['url'].values)
feature_df['label'] = df['label'].values

# Save it
feature_df.to_csv('../data/featurized_dataset.csv', index=False)

print(f"\nSaved featurized dataset: data/featurized_dataset.csv")
print(f"Shape: {feature_df.shape}  ({feature_df.shape[1]-2} numerical features + url + label)")
print("\nColumn names:")
print(list(feature_df.columns))
print("\nFirst few rows:")
print(feature_df.head())