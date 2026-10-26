import pandas as pd

print("Loading Tranco domains (top 30,000 used in training)...")
tranco = pd.read_csv('../data/tranco_top.csv', header=None,
                     names=['rank', 'domain'], nrows=30000)

print(f"Total Tranco domains: {len(tranco)}")

# ── Nulls ─────────────────────────────────────────────────
nulls = tranco['domain'].isnull().sum()
print(f"\nNull domains: {nulls}")

# ── Empty / whitespace ────────────────────────────────────
empty = tranco[tranco['domain'].astype(str).str.strip() == ''].shape[0]
print(f"Empty domains: {empty}")

# ── Malformed (no dot) ────────────────────────────────────
malformed = tranco[~tranco['domain'].astype(str).str.contains(r'\.', regex=True)]
print(f"Malformed domains (no dot): {len(malformed)}")
if len(malformed) > 0:
    print("  Sample:", malformed['domain'].head(5).tolist())

# ── Non-ASCII / weird characters ──────────────────────────
non_ascii = tranco[~tranco['domain'].astype(str).str.match(r'^[\x00-\x7F]+$')]
print(f"Non-ASCII domains: {len(non_ascii)}")
if len(non_ascii) > 0:
    print("  Sample:", non_ascii['domain'].head(5).tolist())

# ── Duplicates ────────────────────────────────────────────
dupes = tranco['domain'].duplicated().sum()
print(f"Duplicate domains: {dupes}")

print("\n" + "="*40)
if nulls == 0 and empty == 0 and len(malformed) == 0 and dupes == 0:
    print("RESULT: Tranco data is clean. No removal needed.")
else:
    print("RESULT: Some issues found — see above.")