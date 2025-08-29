import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from urllib.parse import urlparse

# Define the new, advanced feature extraction function
def extract_advanced_features(url):
    """
    Extracts a comprehensive set of numerical features from a given URL string.
    This function is a core part of the model's intelligence.
    
    Args:
        url (str): The URL to analyze.
        
    Returns:
        dict: A dictionary of advanced features.
    """
    try:
        # A simple check to handle non-URL inputs
        if not isinstance(url, str) or not url.strip():
            return {
                "url_length": 0, "path_length": 0, "hostname_length": 0,
                "has_https": 0, "has_ip": 0, "has_at_symbol": 0,
                "count_digits": 0, "num_subdomains": 0, "is_shortened": 0
            }

        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ""
        path = parsed_url.path or ""

        # Check for common URL shorteners
        shortening_services = ["bit.ly", "goo.gl", "t.co", "ow.ly", "tinyurl.com"]
        is_shortened = int(any(s in hostname for s in shortening_services))

        features = {
            # Total URL length
            "url_length": len(url),
            # Length of the hostname (e.g., google.com)
            "hostname_length": len(hostname),
            # Length of the URL path (e.g., /maps/d/)
            "path_length": len(path),
            # Number of hyphens in the URL
            "num_hyphens": url.count('-'),
            # Number of dots in the URL
            "num_dots": url.count('.'),
            # Presence of 'https'
            "has_https": int(parsed_url.scheme == 'https'),
            # Presence of an IP address instead of a domain name
            "has_ip": int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', hostname))),
            # Presence of an '@' symbol
            "has_at_symbol": int("@" in url),
            # Total number of digits
            "count_digits": sum(c.isdigit() for c in url),
            # Number of subdomains (e.g., mail.google.com has 1)
            "num_subdomains": len(hostname.split('.')) - 2,
            # Flag for URL shorteners
            "is_shortened": is_shortened,
            # Presence of suspicious keywords
            "has_suspicious_words": int(any(word in url.lower() for word in ["login", "verify", "update", "secure", "free", "account", "paypal", "bank", "sign", "insecure", "virus"]))
        }
        return features
    except Exception as e:
        print(f"Error processing URL '{url}': {e}")
        # Return default features in case of an error to prevent the script from crashing
        return {
            "url_length": 0, "path_length": 0, "hostname_length": 0,
            "has_https": 0, "has_ip": 0, "has_at_symbol": 0,
            "count_digits": 0, "num_subdomains": 0, "is_shortened": 0,
            "num_hyphens": 0, "num_dots": 0, "has_suspicious_words": 0
        }

# 1. Load the dataset
# The dataset now contains a variety of safe and suspicious URLs for better training.
df = pd.read_csv("url_data.csv")

# 2. Convert URLs to a DataFrame of features
feature_list = df["url"].apply(extract_advanced_features).tolist()
X = pd.DataFrame(feature_list)
y = df["label"]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model's accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# 6. Save the trained model to a file
dump(model, "model.pkl")
print("✅ Model saved as model.pkl")

