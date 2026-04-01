from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from joblib import load
import re
import math
import pandas as pd
from collections import Counter
from urllib.parse import urlparse, parse_qs
import os

app = Flask(__name__)
CORS(app)

model = load("model.pkl")

# ── Constants (must match train_model.py) ──
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club", ".work",
    ".date", ".racing", ".win", ".bid", ".stream", ".download", ".click",
    ".link", ".info", ".ru", ".cn", ".buzz", ".monster", ".rest", ".icu"
}

SHORTENING_SERVICES = {
    "bit.ly", "goo.gl", "t.co", "ow.ly", "tinyurl.com",
    "rb.gy", "is.gd", "v.gd", "shorte.st", "adf.ly",
    "cutt.ly", "shorturl.at", "tiny.cc"
}

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "free", "account", "paypal",
    "bank", "sign", "insecure", "virus", "malware", "phishing", "confirm",
    "suspend", "restore", "locked", "unlock", "expired", "urgent",
    "winner", "prize", "reward", "claim", "refund", "billing",
    "password", "credential", "authenticate", "ssn", "social-security",
    "tax-refund", "inheritance", "lottery", "bitcoin", "crypto",
    "hack", "crack", "keygen", "torrent", "pirated", "mod-apk",
    "gift-card", "giveaway", "free-money", "earn-money", "cash-online"
]

BRAND_NAMES = [
    "paypal", "apple", "microsoft", "google", "amazon", "netflix",
    "facebook", "instagram", "twitter", "linkedin", "chase", "wellsfargo",
    "citibank", "bankofamerica", "usps", "fedex", "ups", "dhl",
    "irs", "dropbox", "icloud", "onedrive", "yahoo", "outlook"
]


def calculate_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_advanced_features(url):
    try:
        if not isinstance(url, str) or not url.strip():
            return _default_features()

        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ""
        path = parsed_url.path or ""
        query = parsed_url.query or ""
        scheme = parsed_url.scheme or ""

        domain_parts = hostname.split('.') if hostname else []
        tld = ("." + domain_parts[-1]) if domain_parts else ""

        is_shortened = int(any(s in hostname for s in SHORTENING_SERVICES))

        has_brand_impersonation = 0
        for brand in BRAND_NAMES:
            if brand in hostname.lower():
                legit_patterns = [f"{brand}.com", f"{brand}.org", f"{brand}.net",
                                  f"www.{brand}.com", f"{brand}.co"]
                if not any(hostname.lower() == p or hostname.lower().endswith(f".{brand}.com")
                           for p in legit_patterns):
                    has_brand_impersonation = 1
                    break

        leetspeak_patterns = [r'0', r'1', r'3', r'4', r'5', r'7']
        has_leetspeak = int(any(re.search(p, hostname) for p in leetspeak_patterns)
                           and len(hostname) > 5)

        features = {
            "url_length": len(url),
            "hostname_length": len(hostname),
            "path_length": len(path),
            "query_length": len(query),
            "num_hyphens": url.count('-'),
            "num_dots": url.count('.'),
            "num_slashes": url.count('/'),
            "num_underscores": url.count('_'),
            "num_ampersands": url.count('&'),
            "num_equals": url.count('='),
            "num_at_symbols": url.count('@'),
            "num_query_params": len(parse_qs(query)),
            "count_digits": sum(c.isdigit() for c in url),
            "count_special_chars": sum(not c.isalnum() and c not in './-:' for c in url),
            "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
            "letter_ratio": sum(c.isalpha() for c in url) / max(len(url), 1),
            "num_subdomains": max(len(domain_parts) - 2, 0),
            "path_depth": path.count('/') - 1 if path else 0,
            "has_port": int(parsed_url.port is not None and parsed_url.port not in (80, 443)),
            "domain_entropy": round(calculate_entropy(hostname), 4),
            "has_https": int(scheme == 'https'),
            "has_ip": int(bool(re.search(
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', hostname))),
            "has_at_symbol": int("@" in url),
            "is_shortened": is_shortened,
            "has_suspicious_tld": int(tld.lower() in SUSPICIOUS_TLDS),
            "has_double_slash_redirect": int('//' in url[8:]),
            "has_suspicious_words": int(any(
                word in url.lower() for word in SUSPICIOUS_WORDS)),
            "suspicious_word_count": sum(
                1 for word in SUSPICIOUS_WORDS if word in url.lower()),
            "has_brand_impersonation": has_brand_impersonation,
            "has_leetspeak": has_leetspeak,
            "has_exe_or_zip": int(bool(re.search(
                r'\.(exe|zip|scr|bat|cmd|msi|dll|apk|dmg)(\?|$)', url.lower()))),
            "has_redirect_param": int(bool(re.search(
                r'(redirect|next|url|return|goto)=', query.lower()))),
        }
        return features

    except Exception as e:
        print(f"Error: {e}")
        return _default_features()


def _default_features():
    return {
        "url_length": 0, "hostname_length": 0, "path_length": 0, "query_length": 0,
        "num_hyphens": 0, "num_dots": 0, "num_slashes": 0, "num_underscores": 0,
        "num_ampersands": 0, "num_equals": 0, "num_at_symbols": 0, "num_query_params": 0,
        "count_digits": 0, "count_special_chars": 0,
        "digit_ratio": 0, "letter_ratio": 0,
        "num_subdomains": 0, "path_depth": 0, "has_port": 0, "domain_entropy": 0,
        "has_https": 0, "has_ip": 0, "has_at_symbol": 0, "is_shortened": 0,
        "has_suspicious_tld": 0, "has_double_slash_redirect": 0,
        "has_suspicious_words": 0, "suspicious_word_count": 0,
        "has_brand_impersonation": 0, "has_leetspeak": 0,
        "has_exe_or_zip": 0, "has_redirect_param": 0,
    }


def get_risk_flags(features):
    """Generate human-readable risk flags from features."""
    flags = []
    if not features.get("has_https"):
        flags.append({"flag": "No HTTPS", "severity": "high",
                       "detail": "Connection is not encrypted"})
    if features.get("has_ip"):
        flags.append({"flag": "IP Address URL", "severity": "high",
                       "detail": "Uses raw IP instead of domain name"})
    if features.get("has_brand_impersonation"):
        flags.append({"flag": "Brand Impersonation", "severity": "critical",
                       "detail": "URL mimics a known brand name"})
    if features.get("has_leetspeak"):
        flags.append({"flag": "Leetspeak Detected", "severity": "high",
                       "detail": "Domain uses character substitution (e.g., 0 for o)"})
    if features.get("is_shortened"):
        flags.append({"flag": "Shortened URL", "severity": "medium",
                       "detail": "URL uses a shortening service hiding the real destination"})
    if features.get("has_suspicious_tld"):
        flags.append({"flag": "Suspicious TLD", "severity": "medium",
                       "detail": "Domain uses a TLD commonly abused by phishers"})
    if features.get("has_suspicious_words"):
        flags.append({"flag": "Suspicious Keywords", "severity": "medium",
                       "detail": f"URL contains {features.get('suspicious_word_count', 0)} suspicious keyword(s)"})
    if features.get("has_exe_or_zip"):
        flags.append({"flag": "Executable/Archive", "severity": "high",
                       "detail": "URL points to a potentially dangerous file type"})
    if features.get("has_redirect_param"):
        flags.append({"flag": "Redirect Parameter", "severity": "medium",
                       "detail": "URL contains a redirect parameter that could lead elsewhere"})
    if features.get("has_at_symbol"):
        flags.append({"flag": "@ Symbol in URL", "severity": "high",
                       "detail": "@ symbol can be used to deceive about the real destination"})
    if features.get("has_port"):
        flags.append({"flag": "Non-Standard Port", "severity": "medium",
                       "detail": "URL uses a non-standard port number"})
    if features.get("url_length", 0) > 75:
        flags.append({"flag": "Long URL", "severity": "low",
                       "detail": f"URL is {features['url_length']} characters (unusually long)"})
    if features.get("num_hyphens", 0) > 3:
        flags.append({"flag": "Many Hyphens", "severity": "low",
                       "detail": f"URL contains {features['num_hyphens']} hyphens"})
    if features.get("has_double_slash_redirect"):
        flags.append({"flag": "Double Slash Redirect", "severity": "medium",
                       "detail": "URL contains // outside the protocol (possible redirect)"})
    return flags


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/scan", methods=["POST"])
def scan():
    data = request.get_json()
    url = data.get("url") if data else None

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    features = extract_advanced_features(url)
    feature_df = pd.DataFrame([features])

    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]

    # Risk score: probability of being suspicious (0-100)
    risk_score = round(probabilities[1] * 100, 1)

    # Determine risk level
    if risk_score >= 75:
        risk_level = "Critical"
    elif risk_score >= 50:
        risk_level = "High"
    elif risk_score >= 25:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    result = "Suspicious" if prediction == 1 else "Safe"
    risk_flags = get_risk_flags(features)

    return jsonify({
        "result": result,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "flags": risk_flags,
        "details": {
            "url_length": features["url_length"],
            "has_https": bool(features["has_https"]),
            "has_ip_address": bool(features["has_ip"]),
            "is_shortened": bool(features["is_shortened"]),
            "suspicious_tld": bool(features["has_suspicious_tld"]),
            "brand_impersonation": bool(features["has_brand_impersonation"]),
            "num_subdomains": features["num_subdomains"],
            "domain_entropy": features["domain_entropy"],
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
