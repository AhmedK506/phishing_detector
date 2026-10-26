import pandas as pd
from features import extract_features

N_USED = 30000

print(f"Loading top {N_USED} Tranco domains...")
df = pd.read_csv('../data/tranco_top.csv', header=None,
                 names=['rank', 'domain'], nrows=N_USED)

df['url'] = 'https://www.' + df['domain'].astype(str)

print(f"Extracting features from {len(df)} domains...")
features_list = []
for i, url in enumerate(df['url']):
    if i % 5000 == 0:
        print(f"  {i}/{len(df)}...")
    features_list.append(extract_features(str(url)))

feature_df = pd.DataFrame(features_list)
feature_df.insert(0, 'url', df['url'].values)
feature_df['label'] = 0  # all Tranco = safe

feature_df.to_csv('../data/featurized_tranco.csv', index=False)
print(f"\nSaved: featurized_tranco.csv  shape {feature_df.shape}")