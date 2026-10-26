import pandas as pd
import matplotlib.pyplot as plt
import os

# ── Setup 
RAW_PATH = '../data/malicious_phish.csv'
CLEAN_PATH = '../data/cleaned_dataset.csv'
CHART_DIR = '../data/eda_charts'
os.makedirs(CHART_DIR, exist_ok=True)

print("="*60)
print("  DATA PREPROCESSING & EXPLORATORY DATA ANALYSIS")
print("="*60)

# ── 1. Load dataset 
print("\n[1] Loading dataset...")
df = pd.read_csv(RAW_PATH)
print(f"    Total rows: {len(df)}")
print(f"    Columns: {list(df.columns)}")
print(f"    Shape: {df.shape}")

# ── 2. Data types ─────────────────────────────────────────
print("\n[2] Data types:")
print(df.dtypes.to_string())

# ── 3. NULL / missing values ──────────────────────────────
print("\n[3] Checking for NULL / missing values...")
nulls = df.isnull().sum()
print(nulls.to_string())
total_nulls = nulls.sum()
if total_nulls == 0:
    print("    No missing values found.")
else:
    print(f"    Found {total_nulls} missing values - removing them.")
    df = df.dropna()
    print(f"    Rows after removing nulls: {len(df)}")

# ── 4. Empty / whitespace-only URLs ───────────────────────
print("\n[4] Checking for empty / whitespace-only URLs...")
empty_urls = df[df['url'].astype(str).str.strip() == ''].shape[0]
print(f"    Empty URLs: {empty_urls}")
if empty_urls > 0:
    df = df[df['url'].astype(str).str.strip() != '']
    print(f"    Rows after removing empty URLs: {len(df)}")

# ── 5. Malformed URL check ────────────────────────────────
print("\n[5] Checking for malformed URLs (no dot in string)...")
# A valid URL/domain should contain at least one dot
malformed = df[~df['url'].astype(str).str.contains(r'\.', regex=True)]
print(f"    Malformed URLs (no '.'): {len(malformed)}")
if len(malformed) > 0:
    print("    Sample:", malformed['url'].head(3).tolist())
    df = df[df['url'].astype(str).str.contains(r'\.', regex=True)]
    print(f"    Rows after removing malformed URLs: {len(df)}")

# ── 6. DUPLICATES ─────────────────────────────────────────
print("\n[6] Checking for duplicate URLs...")
dup_count = df.duplicated(subset=['url']).sum()
print(f"    Duplicate URLs found: {dup_count}")
if dup_count > 0:
    before = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first')
    print(f"    Removed {before - len(df)} duplicates.")
    print(f"    Rows after removing duplicates: {len(df)}")
else:
    print("    No duplicates found.")

# ── 7. CLASS BALANCE (original 4-class labels) ────────────
print("\n[7] Class distribution (original labels)...")
class_counts = df['type'].value_counts()
print(class_counts.to_string())
print("\n    As percentages:")
class_pct = df['type'].value_counts(normalize=True) * 100
for label, pct in class_pct.items():
    print(f"    {label:12s}: {pct:5.2f}%")

# ── 8. Binary label distribution ──────────────────────────
print("\n[8] Binary distribution (malicious vs safe)...")
def to_binary(label):
    return 0 if label == 'benign' else 1
df['binary_label'] = df['type'].apply(to_binary)
binary_counts = df['binary_label'].value_counts()
safe_n = binary_counts.get(0, 0)
mal_n = binary_counts.get(1, 0)
print(f"    Safe (0):      {safe_n}")
print(f"    Malicious (1): {mal_n}")
safe_pct = safe_n / len(df) * 100
mal_pct = mal_n / len(df) * 100
print(f"    Safe:      {safe_pct:.2f}%")
print(f"    Malicious: {mal_pct:.2f}%")
imbalance_ratio = max(safe_pct, mal_pct) / min(safe_pct, mal_pct)
print(f"    Imbalance ratio: {imbalance_ratio:.2f} : 1")
print("    Note: imbalance is addressed during training via")
print("          class_weight='balanced' in the Random Forest.")

# ── 9. URL length statistics ───────────────────────────
print("\n[9] URL length statistics...")
df['url_length'] = df['url'].astype(str).apply(len)
print(df['url_length'].describe().to_string())

# ── 10. OUTLIER DETECTION (IQR method) ────────────────────
print("\n[10] Outlier detection (IQR method on URL length)...")
Q1 = df['url_length'].quantile(0.25)
Q3 = df['url_length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"    Q1 (25th percentile): {Q1:.0f}")
print(f"    Q3 (75th percentile): {Q3:.0f}")
print(f"    IQR: {IQR:.0f}")
print(f"    Outlier bounds: below {lower_bound:.0f} or above {upper_bound:.0f}")

outliers = df[(df['url_length'] < lower_bound) | (df['url_length'] > upper_bound)]
print(f"    Outliers found: {len(outliers)} ({len(outliers)/len(df)*100:.2f}% of data)")
print(f"    Longest URL: {df['url_length'].max()} characters")
print(f"    Shortest URL: {df['url_length'].min()} characters")

print("\n    Sample of longest URLs (potential outliers):")
for u in df.nlargest(3, 'url_length')['url'].tolist():
    print(f"      [{len(u)} chars] {str(u)[:80]}...")

print("\n    Decision: outliers are RETAINED, not removed.")
print("    Reason: unusually long URLs are often a phishing signal")
print("    (attackers use long/complex URLs to obfuscate), so removing")
print("    them would discard genuine malicious examples.")

# ── 11. Charts for the report ─────────────────────────────
print("\n[11] Generating charts for report...")

plt.figure(figsize=(8, 5))
class_counts.plot(kind='bar', color='#2563eb')
plt.title('Class Distribution (Original Labels)')
plt.xlabel('Class')
plt.ylabel('Number of URLs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/class_distribution.png', dpi=120)
plt.close()
print("    Saved: class_distribution.png")

plt.figure(figsize=(6, 5))
plt.bar(['Safe', 'Malicious'], [safe_n, mal_n], color=['#22c55e', '#ef4444'])
plt.title('Binary Class Balance (Safe vs Malicious)')
plt.ylabel('Number of URLs')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/binary_balance.png', dpi=120)
plt.close()
print("    Saved: binary_balance.png")

plt.figure(figsize=(8, 5))
df[df['url_length'] < 200]['url_length'].hist(bins=50, color='#8b5cf6')
plt.title('URL Length Distribution')
plt.xlabel('URL Length (characters)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/url_length_distribution.png', dpi=120)
plt.close()
print("    Saved: url_length_distribution.png")

plt.figure(figsize=(8, 4))
plt.boxplot(df['url_length'], vert=False)
plt.title('URL Length Distribution with Outliers (Boxplot)')
plt.xlabel('URL Length (characters)')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/url_length_boxplot.png', dpi=120)
plt.close()
print("    Saved: url_length_boxplot.png")

# ── 12. Save cleaned dataset ──────────────────────────────
df_clean = df[['url', 'type']].copy()
df_clean.to_csv(CLEAN_PATH, index=False)
print(f"\n[12] Cleaned dataset saved to: {CLEAN_PATH}")
print(f"     Final row count: {len(df_clean)}")

print("\n" + "="*60)
print("  PREPROCESSING COMPLETE")
print("="*60)