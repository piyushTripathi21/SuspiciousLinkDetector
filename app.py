from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import re
import pandas as pd
from urllib.parse import urlparse
import os   

app = Flask(__name__)
CORS(app)

model = load("model.pkl")

def extract_advanced_features(url):
    try:
        if not isinstance(url, str) or not url.strip():
            return {
                "url_length": 0, "path_length": 0, "hostname_length": 0,
                "has_https": 0, "has_ip": 0, "has_at_symbol": 0,
                "count_digits": 0, "num_subdomains": 0, "is_shortened": 0,
                "num_hyphens": 0, "num_dots": 0, "has_suspicious_words": 0
            }

        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ""
        path = parsed_url.path or ""

        shortening_services = ["bit.ly", "goo.gl", "t.co", "ow.ly", "tinyurl.com"]
        is_shortened = int(any(s in hostname for s in shortening_services))

        features = {
            "url_length": len(url),
            "hostname_length": len(hostname),
            "path_length": len(path),
            "num_hyphens": url.count('-'),
            "num_dots": url.count('.'),
            "has_https": int(parsed_url.scheme == 'https'),
            "has_ip": int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', hostname))),
            "has_at_symbol": int("@" in url),
            "count_digits": sum(c.isdigit() for c in url),
            "num_subdomains": max(len(hostname.split('.')) - 2, 0),
            "is_shortened": is_shortened,
            "has_suspicious_words": int(any(word in url.lower() for word in [
                "login", "verify", "update", "secure", "free",
                "account", "paypal", "bank", "sign", "insecure", "virus"
            ]))
        }
        return features

    except Exception as e:
        print(e)
        return {k: 0 for k in [
            "url_length", "hostname_length", "path_length", "num_hyphens",
            "num_dots", "has_https", "has_ip", "has_at_symbol",
            "count_digits", "num_subdomains", "is_shortened", "has_suspicious_words"
        ]}


@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Suspicious Link Detector API is running"
    })

@app.route("/scan", methods=["POST"])
def scan():
    data = request.get_json()
    url = data.get("url") if data else None

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    parsed_url = urlparse(url)
    if parsed_url.scheme == "http":
        return jsonify({"result": "Suspicious (No HTTPS found)"}), 200

    features = extract_advanced_features(url)
    feature_df = pd.DataFrame([features])
    prediction = model.predict(feature_df)[0]

    result = "Suspicious" if prediction == 1 else "Safe"
    return jsonify({"result": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
