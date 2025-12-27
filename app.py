from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import re
import pandas as pd
from urllib.parse import urlparse

# Initialize the Flask app
app = Flask(__name__)
# Enable CORS to allow the frontend to access the backend
CORS(app)

# Load the trained model from the pkl file
model = load("model.pkl")

# Define the feature extraction function, identical to train_model.py
def extract_advanced_features(url):
    """
    Extracts a comprehensive set of numerical features from a given URL string.
    """
    try:
        if not isinstance(url, str) or not url.strip():
            return {
                "url_length": 0, "path_length": 0, "hostname_length": 0,
                "has_https": 0, "has_ip": 0, "has_at_symbol": 0,
                "count_digits": 0, "num_subdomains": 0, "is_shortened": 0
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
            "num_subdomains": len(hostname.split('.')) - 2,
            "is_shortened": is_shortened,
            "has_suspicious_words": int(any(word in url.lower() for word in [
                "login", "verify", "update", "secure", "free", "account", "paypal", "bank", "sign", "insecure", "virus"
            ]))
        }
        return features
    except Exception as e:
        print(f"Error processing URL '{url}': {e}")
        return {
            "url_length": 0, "path_length": 0, "hostname_length": 0,
            "has_https": 0, "has_ip": 0, "has_at_symbol": 0,
            "count_digits": 0, "num_subdomains": 0, "is_shortened": 0,
            "num_hyphens": 0, "num_dots": 0, "has_suspicious_words": 0
        }

# Define the API endpoint to scan URLs
@app.route("/scan", methods=["POST"])
def scan():
    # Get the URL from the JSON request body
    data = request.get_json()
    url = data.get("url")

    # Handle cases where no URL is provided
    if not url:
        return jsonify({"result": "Error: No URL provided"}), 400

    # --- Manual Rule for HTTP (without HTTPS) ---
    parsed_url = urlparse(url)
    if parsed_url.scheme == "http":
        return jsonify({"result": "Suspicious (No HTTPS found)"}), 200

    # Extract advanced features from the input URL
    features = extract_advanced_features(url)
    
    # Convert the features dictionary into a pandas DataFrame
    feature_df = pd.DataFrame([features])
    
    # Make a prediction using the loaded model
    prediction = model.predict(feature_df)[0]

    # Map the numerical prediction to a readable result string
    result = "Suspicious" if prediction == 1 else "Safe"
    
    # Return the result as a JSON response
    return jsonify({"result": result})

# Run the Flask app in debug mode
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    #app.run(debug=True)
