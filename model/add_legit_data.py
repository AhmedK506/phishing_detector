import pandas as pd
import csv

# ── Config ────────────────────────────────────────────────
TRANCO_PATH = '../data/tranco_top.csv'
ORIGINAL_PATH = '../data/cleaned_dataset.csv'
OUTPUT_PATH = '../data/combined_dataset.csv'
NUM_LEGIT = 30000

# ── 1. Load original dataset ──────────────────────────────
print("Loading original dataset...")
original = pd.read_csv(ORIGINAL_PATH)
print(f"  Original rows: {len(original)}")
print(original['type'].value_counts())

# ── 2. Load Tranco list (tab-separated, no header) ────────
print(f"\nLoading top {NUM_LEGIT} Tranco domains...")
tranco = pd.read_csv(
    TRANCO_PATH,
    sep=',',
    header=None,
    names=['rank', 'domain'],
    nrows=NUM_LEGIT,
    quoting=csv.QUOTE_NONE
)
print(f"  Loaded {len(tranco)} domains")
print("  Sample:", tranco['domain'].head(3).tolist())

# ── 3. Turn domains into realistic URLs, label as benign ──
legit_urls = []
for domain in tranco['domain']:
    domain = str(domain).strip()
    if domain and '.' in domain:
        legit_urls.append({
            'url': f'https://www.{domain}',
            'type': 'benign'
        })

legit_df = pd.DataFrame(legit_urls)
print(f"\n  Built {len(legit_df)} legitimate URL rows")

# ── Safety check: make sure we actually built legit rows ──
if len(legit_df) < 1000:
    print("\n❌ ERROR: Almost no legit rows were built — domains didn't parse.")
    print("   Aborting so we don't overwrite with bad data.")
    exit()

# ── 4. Combine and remove duplicates ──────────────────────
combined = pd.concat([original, legit_df], ignore_index=True)
before = len(combined)
combined = combined.drop_duplicates(subset=['url'], keep='first')
print(f"\nCombined rows: {len(combined)} (removed {before - len(combined)} duplicate URLs)")
print(combined['type'].value_counts())

# ── 5. Save ───────────────────────────────────────────────
combined.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved combined dataset to {OUTPUT_PATH}")