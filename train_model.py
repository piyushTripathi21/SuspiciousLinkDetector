import pandas as pd
import numpy as np
import re
import math
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
from urllib.parse import urlparse, parse_qs


# ── Suspicious TLDs commonly used by phishing/malware sites ──
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",  # Free TLDs abused by phishers
    ".xyz", ".top", ".club", ".work", ".date", ".racing", ".win",
    ".bid", ".stream", ".download", ".click", ".link", ".info",
    ".ru", ".cn", ".buzz", ".monster", ".rest", ".icu"
}

# ── Known URL shorteners ──
SHORTENING_SERVICES = {
    "bit.ly", "goo.gl", "t.co", "ow.ly", "tinyurl.com",
    "rb.gy", "is.gd", "v.gd", "shorte.st", "adf.ly",
    "cutt.ly", "shorturl.at", "tiny.cc"
}

# ── Suspicious keywords in URLs ──
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

# ── Legitimate brand names (used to detect impersonation) ──
BRAND_NAMES = [
    "paypal", "apple", "microsoft", "google", "amazon", "netflix",
    "facebook", "instagram", "twitter", "linkedin", "chase", "wellsfargo",
    "citibank", "bankofamerica", "usps", "fedex", "ups", "dhl",
    "irs", "dropbox", "icloud", "onedrive", "yahoo", "outlook"
]


def calculate_entropy(text):
    """Calculate Shannon entropy of a string — higher entropy means more randomness."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_advanced_features(url):
    """
    Extracts 25+ numerical features from a URL for phishing/malware detection.
    """
    try:
        if not isinstance(url, str) or not url.strip():
            return _default_features()

        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ""
        path = parsed_url.path or ""
        query = parsed_url.query or ""
        scheme = parsed_url.scheme or ""

        # Domain parts
        domain_parts = hostname.split('.') if hostname else []
        tld = ("." + domain_parts[-1]) if domain_parts else ""

        # Check for URL shorteners
        is_shortened = int(any(s in hostname for s in SHORTENING_SERVICES))

        # Check for brand impersonation (brand in domain but not the real domain)
        has_brand_impersonation = 0
        for brand in BRAND_NAMES:
            if brand in hostname.lower():
                # If the brand is in the hostname but the hostname is NOT the real domain
                legit_patterns = [f"{brand}.com", f"{brand}.org", f"{brand}.net",
                                  f"www.{brand}.com", f"{brand}.co"]
                if not any(hostname.lower() == p or hostname.lower().endswith(f".{brand}.com") for p in legit_patterns):
                    has_brand_impersonation = 1
                    break

        # Detect leetspeak substitutions (e.g., paypa1, g00gle, amaz0n)
        leetspeak_patterns = [r'0', r'1', r'3', r'4', r'5', r'7']
        has_leetspeak = int(any(re.search(p, hostname) for p in leetspeak_patterns)
                           and len(hostname) > 5)

        features = {
            # ── Length-based features ──
            "url_length": len(url),
            "hostname_length": len(hostname),
            "path_length": len(path),
            "query_length": len(query),

            # ── Count-based features ──
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

            # ── Ratio features ──
            "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
            "letter_ratio": sum(c.isalpha() for c in url) / max(len(url), 1),

            # ── Domain features ──
            "num_subdomains": max(len(domain_parts) - 2, 0),
            "path_depth": path.count('/') - 1 if path else 0,
            "has_port": int(parsed_url.port is not None and parsed_url.port not in (80, 443)),
            "domain_entropy": round(calculate_entropy(hostname), 4),

            # ── Security features ──
            "has_https": int(scheme == 'https'),
            "has_ip": int(bool(re.search(
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', hostname))),
            "has_at_symbol": int("@" in url),
            "is_shortened": is_shortened,
            "has_suspicious_tld": int(tld.lower() in SUSPICIOUS_TLDS),
            "has_double_slash_redirect": int('//' in url[8:]),

            # ── Content-analysis features ──
            "has_suspicious_words": int(any(
                word in url.lower() for word in SUSPICIOUS_WORDS)),
            "suspicious_word_count": sum(
                1 for word in SUSPICIOUS_WORDS if word in url.lower()),
            "has_brand_impersonation": has_brand_impersonation,
            "has_leetspeak": has_leetspeak,

            # ── File/extension features ──
            "has_exe_or_zip": int(bool(re.search(
                r'\.(exe|zip|scr|bat|cmd|msi|dll|apk|dmg)(\?|$)', url.lower()))),
            "has_redirect_param": int(bool(re.search(
                r'(redirect|next|url|return|goto)=', query.lower()))),
        }
        return features

    except Exception as e:
        print(f"Error processing URL '{url}': {e}")
        return _default_features()


def _default_features():
    """Returns a dictionary of all features set to 0."""
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


def main():
    # 1. Load the dataset
    df = pd.read_csv("url_data.csv")
    print(f"📊 Dataset loaded: {len(df)} URLs ({df['label'].value_counts().to_dict()})")

    # 2. Extract features
    feature_list = df["url"].apply(extract_advanced_features).tolist()
    X = pd.DataFrame(feature_list)
    y = df["label"]

    print(f"🔢 Features extracted: {X.shape[1]} features per URL")

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train with optimized hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"📈 Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Suspicious"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"  True Safe: {cm[0][0]}  |  False Suspicious: {cm[0][1]}")
    print(f"  False Safe: {cm[1][0]}  |  True Suspicious: {cm[1][1]}")

    # Feature importance
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n🏆 Top 10 Most Important Features:")
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # 6. Save model
    dump(model, "model.pkl")
    print(f"\n✅ Model saved as model.pkl")


if __name__ == "__main__":
    main()

