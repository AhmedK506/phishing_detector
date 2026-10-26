import requests
from bs4 import BeautifulSoup
import re

# Request headers to mimic a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36'
}

def analyze_html(url):
    result = {
        'success': False,
        'risk_score': 0,
        'risk_level': 'Unknown',
        'indicators': [],
        'features': {}
    }

    # ── Fetch page content ────────────────────────────────
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=8,
            verify=False,  # some phishing sites have bad SSL
            allow_redirects=True
        )
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        result['success'] = True

    except Exception as e:
        result['error'] = str(e)
        result['risk_score'] = 30  # unknown = slightly suspicious
        result['risk_level'] = 'Unknown'
        return result

    score = 0
    indicators = []
    features = {}

    # ── 1. Form analysis ──────────────────────────────────
    forms = soup.find_all('form')
    features['num_forms'] = len(forms)

    external_form_actions = 0
    for form in forms:
        action = form.get('action', '')
        if action.startswith('http') and url.split('/')[2] not in action:
            external_form_actions += 1

    features['external_form_actions'] = external_form_actions
    if external_form_actions > 0:
        score += 30
        indicators.append(f'Form submits to external domain ({external_form_actions} found)')

    # ── 2. iFrame analysis ────────────────────────────────
    iframes = soup.find_all('iframe')
    features['num_iframes'] = len(iframes)
    hidden_iframes = [
        i for i in iframes
        if i.get('style') and ('display:none' in i.get('style', '').replace(' ', '')
        or 'visibility:hidden' in i.get('style', '').replace(' ', ''))
    ]
    features['num_hidden_iframes'] = len(hidden_iframes)

    if len(hidden_iframes) > 0:
        score += 20
        indicators.append(f'Hidden iframes detected ({len(hidden_iframes)} found)')

    if len(iframes) > 3:
        score += 10
        indicators.append(f'Excessive iframes ({len(iframes)} found)')

    # ── 3. Link analysis ──────────────────────────────────
    all_links = soup.find_all('a', href=True)
    external_links = [
        l for l in all_links
        if l['href'].startswith('http')
        and url.split('/')[2] not in l['href']
    ]
    features['num_links'] = len(all_links)
    features['num_external_links'] = len(external_links)

    if len(all_links) > 0:
        external_ratio = len(external_links) / len(all_links)
        features['external_link_ratio'] = round(external_ratio, 2)
        if external_ratio > 0.7:
            score += 20
            indicators.append(f'High external link ratio ({round(external_ratio*100)}%)')
    else:
        features['external_link_ratio'] = 0

    # ── 4. JavaScript analysis ────────────────────────────
    scripts = soup.find_all('script')
    js_content = ' '.join([s.string or '' for s in scripts])

    suspicious_js = [
        'eval(', 'unescape(', 'document.write(',
        'window.location', 'document.cookie',
        'atob(', 'fromCharCode('
    ]
    suspicious_js_found = [s for s in suspicious_js if s in js_content]
    features['num_scripts'] = len(scripts)
    features['suspicious_js_count'] = len(suspicious_js_found)

    if len(suspicious_js_found) >= 2:
        score += 15
        indicators.append(f'Suspicious JS functions: {", ".join(suspicious_js_found)}')

    # ── 5. Password field without HTTPS ──────────────────
    password_fields = soup.find_all('input', {'type': 'password'})
    features['has_password_field'] = 1 if password_fields else 0

    if password_fields and not url.startswith('https'):
        score += 25
        indicators.append('Password field on non-HTTPS page')

    # ── 6. Favicon from different domain ─────────────────
    favicon = soup.find('link', rel=lambda r: r and 'icon' in r)
    features['has_external_favicon'] = 0
    if favicon and favicon.get('href', '').startswith('http'):
        if url.split('/')[2] not in favicon['href']:
            features['has_external_favicon'] = 1
            score += 10
            indicators.append('Favicon loaded from external domain')

    # ── 7. Title analysis ─────────────────────────────────
    title = soup.find('title')
    title_text = title.string if title else ''
    features['has_title'] = 1 if title_text else 0

    suspicious_title_words = [
        'login', 'verify', 'account', 'secure',
        'update', 'confirm', 'banking'
    ]
    if any(w in title_text.lower() for w in suspicious_title_words):
        score += 10
        indicators.append(f'Suspicious page title: "{title_text}"')

    # ── 8. Disable right click ────────────────────────────
    if 'contextmenu' in js_content and 'return false' in js_content:
        score += 10
        indicators.append('Right-click disabled (common phishing tactic)')

    # ── Cap score at 100 ──────────────────────────────────
    score = min(score, 100)

    # ── Risk level ────────────────────────────────────────
    if score >= 60:
        risk_level = 'High'
    elif score >= 30:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'

    result['risk_score'] = score
    result['risk_level'] = risk_level
    result['indicators'] = indicators
    result['features'] = features

    return result