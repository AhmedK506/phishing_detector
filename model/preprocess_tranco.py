import pandas as pd
import matplotlib.pyplot as plt
import os

TRANCO_PATH = '../data/tranco_top.csv'
CHART_DIR = '../data/eda_charts'
N_USED = 30000  # the number you actually used in training
os.makedirs(CHART_DIR, exist_ok=True)

print("="*60)
print("  TRANCO DATASET - PREPROCESSING & EDA")
print("="*60)

# ── 1. Load (only the top N you actually used) ────────────
print(f"\n[1] Loading top {N_USED} Tranco domains...")
df = pd.read_csv(TRANCO_PATH, header=None, names=['rank', 'domain'], nrows=N_USED)
print(f"    Rows: {len(df)}")
print(f"    Columns: {list(df.columns)}")

# ── 2. Data types ─────────────────────────────────────────
print("\n[2] Data types:")
print(df.dtypes.to_string())

# ── 3. Nulls ──────────────────────────────────────────────
print("\n[3] Null / missing values...")
print(df.isnull().sum().to_string())
print(f"    Total nulls: {df.isnull().sum().sum()}")

# ── 4. Empty domains ──────────────────────────────────────
print("\n[4] Empty / whitespace domains...")
empty = df[df['domain'].astype(str).str.strip() == ''].shape[0]
print(f"    Empty: {empty}")

# ── 5. Malformed (no dot) ─────────────────────────────────
print("\n[5] Malformed domains (no dot)...")
malformed = df[~df['domain'].astype(str).str.contains(r'\.', regex=True)]
print(f"    Malformed: {len(malformed)}")
if len(malformed) > 0:
    print("    Sample:", malformed['domain'].head(5).tolist())

# ── 6. Non-ASCII / IDN domains ────────────────────────────
print("\n[6] Non-ASCII / internationalized domains...")
non_ascii = df[~df['domain'].astype(str).str.match(r'^[\x00-\x7F]+$')]
print(f"    Non-ASCII: {len(non_ascii)}")
if len(non_ascii) > 0:
    print("    Sample:", non_ascii['domain'].head(5).tolist())

# ── 7. Duplicates ─────────────────────────────────────────
print("\n[7] Duplicate domains...")
dupes = df['domain'].duplicated().sum()
print(f"    Duplicates: {dupes}")

# ── 8. Domain length statistics ───────────────────────────
print("\n[8] Domain length statistics...")
df['domain_length'] = df['domain'].astype(str).apply(len)
print(df['domain_length'].describe().to_string())

# ── 9. Outlier detection (IQR) on domain length ───────────
print("\n[9] Outlier detection (IQR) on domain length...")
Q1 = df['domain_length'].quantile(0.25)
Q3 = df['domain_length'].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers = df[(df['domain_length'] < lower) | (df['domain_length'] > upper)]
print(f"    Q1={Q1:.0f}  Q3={Q3:.0f}  IQR={IQR:.0f}")
print(f"    Bounds: below {lower:.0f} or above {upper:.0f}")
print(f"    Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
print(f"    Longest domain: {df['domain_length'].max()} chars")
print(f"    Shortest domain: {df['domain_length'].min()} chars")

# ── 10. TLD distribution ──────────────────────────────────
print("\n[10] Top 10 TLDs (domain endings)...")
df['tld'] = df['domain'].astype(str).str.split('.').str[-1]
print(df['tld'].value_counts().head(10).to_string())

# ── 11. Charts ────────────────────────────────────────────
print("\n[11] Generating charts...")

plt.figure(figsize=(8,5))
df['domain_length'].hist(bins=40, color='#2563eb')
plt.title('Tranco Domain Length Distribution')
plt.xlabel('Domain Length (characters)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/tranco_domain_length.png', dpi=120)
plt.close()
print("    Saved: tranco_domain_length.png")

plt.figure(figsize=(8,5))
df['tld'].value_counts().head(10).plot(kind='bar', color='#8b5cf6')
plt.title('Tranco - Top 10 TLDs')
plt.xlabel('TLD')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{CHART_DIR}/tranco_tld_distribution.png', dpi=120)
plt.close()
print("    Saved: tranco_tld_distribution.png")

print("\n" + "="*60)
print("  TRANCO PREPROCESSING COMPLETE")
print("="*60)