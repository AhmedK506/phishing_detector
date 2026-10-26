import re
import tldextract
from urllib.parse import urlparse

# Known legitimate domains that should never be flagged
WHITELIST = [
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
    'instagram.com', 'linkedin.com', 'microsoft.com', 'apple.com',
    'amazon.com', 'wikipedia.org', 'github.com', 'stackoverflow.com',
    'reddit.com', 'netflix.com', 'spotify.com', 'dropbox.com'
]

SHORTENING_SERVICES = [
    'bit.ly', 'tinyurl', 'goo.gl', 't.co',
    'ow.ly', 'is.gd', 'buff.ly', 'shorte.st'
]

SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'update', 'secure',
    'account', 'banking', 'confirm', 'password', 'free',
    'lucky', 'bonus', 'click', 'prize', 'winner', 'paypal',
    'ebay', 'amazon', 'apple', 'microsoft', 'support'
]

def extract_features(url):
    features = {}
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    domain = ext.domain
    suffix = ext.suffix
    subdomain = ext.subdomain
    path = parsed.path
    query = parsed.query

    # ── URL length features ───────────────────────────────
    features['url_length'] = len(url)
    features['domain_length'] = len(domain)
    features['path_length'] = len(path)
    features['query_length'] = len(query)
    features['subdomain_length'] = len(subdomain)

    # ── Character count features ──────────────────────────
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_question_marks'] = url.count('?')
    features['num_equal_signs'] = url.count('=')
    features['num_at_signs'] = url.count('@')
    features['num_percent'] = url.count('%')
    features['num_hash'] = url.count('#')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_digits_in_domain'] = sum(c.isdigit() for c in domain)
    features['num_path_segments'] = path.count('/')

    # ── Domain features ───────────────────────────────────
    features['has_subdomain'] = 1 if subdomain else 0
    features['is_https'] = 1 if parsed.scheme == 'https' else 0
    features['domain_has_hyphen'] = 1 if '-' in domain else 0
    features['domain_has_digits'] = 1 if any(c.isdigit() for c in domain) else 0

    # ── IP address check ──────────────────────────────────
    host = parsed.netloc.split(':')[0]
    features['has_ip'] = 1 if re.match(
        r'^(\d{1,3}\.){3}\d{1,3}$', host) else 0

    # ── Suspicious patterns ───────────────────────────────
    features['suspicious_keyword_count'] = sum(
        1 for word in SUSPICIOUS_KEYWORDS if word in url.lower()
    )
    features['is_shortened'] = 1 if any(
        s in url for s in SHORTENING_SERVICES) else 0

    # Brand name in subdomain/path (not in domain) = suspicious
    brands = ['paypal', 'apple', 'microsoft', 'google',
              'amazon', 'facebook', 'netflix', 'ebay']
    features['brand_in_subdomain'] = 1 if any(
        b in subdomain.lower() for b in brands) else 0

    # ── Ratio features ────────────────────────────────────
    features['digit_ratio'] = features['num_digits'] / \
        len(url) if len(url) > 0 else 0
    features['special_char_ratio'] = (
        features['num_dots'] + features['num_hyphens'] +
        features['num_slashes']
    ) / len(url) if len(url) > 0 else 0

    # ── Path-based impersonation features ─────────────────
    path_lower = path.lower()
    full_lower = url.lower()

    # Brand name appearing in the PATH (not the legit domain) — classic impersonation
    brands = ['paypal', 'apple', 'microsoft', 'google', 'amazon',
              'facebook', 'netflix', 'ebay', 'skype', 'outlook',
              'instagram', 'whatsapp', 'bank', 'secure', 'account']
    features['brand_in_path'] = sum(1 for b in brands if b in path_lower)

    # Suspicious path keywords often seen in phishing kits
    phishing_path_words = ['cgi-bin', 'verification', 'verify', 'webscr',
                           'signin', 'login', 'update', 'confirm',
                           'secure', 'account', 'suspend', 'unlock']
    features['suspicious_path_keywords'] = sum(
        1 for w in phishing_path_words if w in path_lower)

    # An embedded domain-like string inside the path (e.g. "login.skype.com" sitting in the path)
    # Counts occurrences of things that look like "word.word.tld" in the path
    embedded_domains = re.findall(
        r'[a-z0-9-]+\.[a-z0-9-]+\.(?:com|net|org|info|biz|co)', path_lower)
    features['embedded_domain_in_path'] = len(embedded_domains)

    # Long hexadecimal tokens (random hashes common in phishing URLs)
    hex_tokens = re.findall(r'[a-f0-9]{16,}', full_lower)
    features['has_long_hex_token'] = 1 if hex_tokens else 0
    features['num_long_hex_tokens'] = len(hex_tokens)

    # Count of subdomains/dots in the path portion specifically
    features['dots_in_path'] = path.count('.')

    # ── Content/semantic features (target digit-free & clean-domain phishing) ──
    import math

    # Domain entropy — random-looking domains score high even without digits
    def shannon_entropy(s):
        if not s:
            return 0
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * math.log2(p) for p in probs)
    features['domain_entropy'] = round(shannon_entropy(domain), 3)

    # Suspicious TLDs — cheap TLDs heavily abused by phishing
    suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work',
                       'click', 'link', 'info', 'buzz', 'live', 'icu', 'rest']
    features['suspicious_tld'] = 1 if suffix.split('.')[-1] in suspicious_tlds else 0

    # Consonant-to-vowel ratio in domain — random strings have odd ratios
    vowels = sum(1 for c in domain.lower() if c in 'aeiou')
    consonants = sum(1 for c in domain.lower() if c.isalpha() and c not in 'aeiou')
    features['domain_consonant_ratio'] = round(
        consonants / (vowels + 1), 3)  # +1 avoids divide-by-zero

    # Longest consecutive consonant run — gibberish domains have long runs
    max_consonant_run = 0
    current_run = 0
    for c in domain.lower():
        if c.isalpha() and c not in 'aeiou':
            current_run += 1
            max_consonant_run = max(max_consonant_run, current_run)
        else:
            current_run = 0
    features['max_consonant_run'] = max_consonant_run

    # Is a "normal" legit subdomain (reduces false positives on legit sites)
    normal_subdomains = ['www', 'mail', 'accounts', 'login', 'm',
                         'shop', 'store', 'app', 'my', 'secure', 'portal']
    features['is_normal_subdomain'] = 1 if subdomain.lower() in normal_subdomains else 0

    return features