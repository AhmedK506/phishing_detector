import pandas as pd
import tldextract

print("Loading datasets...")
kaggle = pd.read_csv('../data/cleaned_dataset.csv')
tranco = pd.read_csv('../data/tranco_top.csv', header=None,
                     names=['rank', 'domain'], nrows=30000)

def encode_label(label):
    if label == 'phishing': return 1
    elif label == 'malware': return 1
    else: return 0
kaggle['label'] = kaggle['type'].apply(encode_label)

print(f"  Kaggle rows: {len(kaggle)}")
print(f"  Tranco rows: {len(tranco)}")

# ── 1. Same URL, different labels ─────────────────────────
print("\n" + "="*60)
print("  CHECK 1: Same URL with conflicting labels")
print("="*60)

url_label_counts = kaggle.groupby('url')['label'].nunique()
conflicting_urls = url_label_counts[url_label_counts > 1]
print(f"\n  URLs with more than one label: {len(conflicting_urls)}")

if len(conflicting_urls) > 0:
    print("\n  Sample conflicts:")
    for u in conflicting_urls.index[:5]:
        rows = kaggle[kaggle['url'] == u]
        print(f"    {str(u)[:60]}")
        print(f"      labels: {rows['type'].tolist()}")
else:
    print("  No URL-level label conflicts found.")

# ── 2. Same domain, conflicting labels ────────────────────
print("\n" + "="*60)
print("  CHECK 2: Same domain with conflicting labels")
print("="*60)

print("\n  Extracting domains (this takes a moment)...")
def get_domain(url):
    ext = tldextract.extract(str(url))
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

kaggle['domain'] = kaggle['url'].apply(get_domain)

domain_labels = kaggle.groupby('domain')['label'].nunique()
mixed_domains = domain_labels[domain_labels > 1]
print(f"\n  Domains with both safe and malicious URLs: {len(mixed_domains)}")
print(f"  As a percentage of unique domains: "
      f"{len(mixed_domains)/kaggle['domain'].nunique()*100:.2f}%")

if len(mixed_domains) > 0:
    print("\n  Sample mixed domains (top 5 by frequency):")
    for d in mixed_domains.index[:5]:
        rows = kaggle[kaggle['domain'] == d]
        safe_n = (rows['label'] == 0).sum()
        mal_n = (rows['label'] == 1).sum()
        print(f"    {d:35s}  safe: {safe_n:4d}  malicious: {mal_n:4d}")
    print("\n  Note: mixed labels at domain level are not necessarily errors,")
    print("  as legitimate domains can host compromised or malicious pages.")

# ── 3. Cross-source conflicts (Tranco vs Kaggle) ──────────
print("\n" + "="*60)
print("  CHECK 3: Tranco 'safe' domains appearing as phishing in Kaggle")
print("="*60)

tranco_domains = set(tranco['domain'].astype(str).str.lower())
kaggle_malicious_domains = set(
    kaggle[kaggle['label'] == 1]['domain'].astype(str).str.lower()
)

conflicts = tranco_domains & kaggle_malicious_domains
print(f"\n  Conflicting domains: {len(conflicts)}")

if len(conflicts) > 0:
    print("\n  Sample (labelled safe in Tranco, malicious in Kaggle):")
    for d in list(conflicts)[:10]:
        n = len(kaggle[(kaggle['domain'].str.lower() == d) & (kaggle['label'] == 1)])
        print(f"    {d:35s}  ({n} malicious URLs in Kaggle)")
    print("\n  These represent contradictory labelling between sources.")
else:
    print("  No cross-source conflicts found.")

print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
print(f"  URL-level conflicts:     {len(conflicting_urls)}")
print(f"  Domain-level mixed:      {len(mixed_domains)}")
print(f"  Cross-source conflicts:  {len(conflicts)}")