from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import imagehash
from PIL import Image
import os
import io
import time

# Pre-computed hashes of known brand homepages
# We'll generate these on first run
BRAND_HASHES_FILE = 'model/brand_hashes.pkl'

KNOWN_BRANDS = {
    'paypal': 'https://www.paypal.com',
    'facebook': 'https://www.facebook.com',
    'google': 'https://www.google.com',
    'microsoft': 'https://www.microsoft.com',
    'apple': 'https://www.apple.com',
    'amazon': 'https://www.amazon.com',
    'netflix': 'https://www.netflix.com',
    'instagram': 'https://www.instagram.com',
}

def get_driver():
    options = Options()
    options.add_argument('--headless')       # run without opening browser
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1280,800')
    options.add_argument('--disable-gpu')
    options.add_argument('--log-level=3')   # suppress logs

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def take_screenshot(url, driver=None):
    close_driver = False
    if driver is None:
        driver = get_driver()
        close_driver = True
    try:
        driver.get(url)
        time.sleep(3)  # wait for page to load
        screenshot = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(screenshot))
        return img
    except Exception as e:
        return None
    finally:
        if close_driver:
            driver.quit()

def get_image_hash(img):
    return imagehash.phash(img)

def generate_brand_hashes():
    import joblib
    print("Generating brand hashes (this runs once)...")
    driver = get_driver()
    brand_hashes = {}

    for brand, url in KNOWN_BRANDS.items():
        print(f"  Capturing {brand}...")
        img = take_screenshot(url, driver)
        if img:
            brand_hashes[brand] = get_image_hash(img)
            print(f"  ✓ {brand} captured")
        else:
            print(f"  ✗ {brand} failed")

    driver.quit()
    joblib.dump(brand_hashes, BRAND_HASHES_FILE)
    print(f"Brand hashes saved to {BRAND_HASHES_FILE}")
    return brand_hashes

def load_brand_hashes():
    import joblib
    if os.path.exists(BRAND_HASHES_FILE):
        return joblib.load(BRAND_HASHES_FILE)
    return generate_brand_hashes()

def analyze_visual(url):
    result = {
        'success': False,
        'risk_score': 0,
        'risk_level': 'Unknown',
        'indicators': [],
        'closest_brand': None,
        'similarity_score': 0
    }

    try:
        # Take screenshot of target URL
        img = take_screenshot(url)
        if img is None:
            result['error'] = 'Could not capture screenshot'
            return result

        result['success'] = True
        url_hash = get_image_hash(img)

        # Load brand hashes and compare
        brand_hashes = load_brand_hashes()
        score = 0
        indicators = []
        min_distance = 100
        closest_brand = None

        import tldextract
        import re
        from urllib.parse import urlparse

        # Get the host portion of the URL (handles ports too)
        host = urlparse(url).hostname or ''

        # If the host is a bare IP address, use it as-is
        if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', host):
            url_domain = host
        else:
            ext = tldextract.extract(url)
            url_domain = f"{ext.domain}.{ext.suffix}"

        for brand, brand_hash in brand_hashes.items():
            distance = url_hash - brand_hash
            similarity = max(0, 100 - (distance * 1.5))

            if distance < min_distance:
                min_distance = distance
                closest_brand = brand
                best_similarity = similarity

        result['closest_brand'] = closest_brand
        result['similarity_score'] = round(best_similarity, 2)

        # If visually similar to a brand but domain doesn't match
        if best_similarity > 80:
            # The brand's true registered domain, e.g. "paypal.com"
            brand_domain = KNOWN_BRANDS.get(closest_brand, '')
            brand_ext = tldextract.extract(brand_domain)
            brand_registered = f"{brand_ext.domain}.{brand_ext.suffix}"

            # Legitimate ONLY if the page's registered domain exactly matches
            if url_domain == brand_registered:
                score = 0  # genuine brand on its real domain
            else:
                score = min(100, int(best_similarity))
                indicators.append(
                    f'Page layout resembles {closest_brand} ({round(best_similarity)}% visual similarity) '
                    f'but is hosted on {url_domain}, not the official {brand_registered} domain '
                    f'— possible brand impersonation'
                )

        # Risk level
        if score >= 60:
            risk_level = 'High'
        elif score >= 30:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        result['risk_score'] = score
        result['risk_level'] = risk_level
        result['indicators'] = indicators

    except Exception as e:
        result['error'] = str(e)

    return result