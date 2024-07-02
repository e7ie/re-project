#BONUS
import requests
import sys
import json

# URL of the Flask API for the basic features endpoint
url = "http://127.0.0.1:5010/v2/predict_basic"

# ONLY basic features
data = {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 1800,
    "sqft_basement": 0,
    "yr_built": 1990,
    "yr_renovated": 0,
    "lat": 47.6673,
    "long": -122.3212,
    "sqft_living15": 1800,
    "sqft_lot15": 5000
}

print(f"Sending request to {url}")
sys.stdout.flush()

# Sending the POST request with JSON data
try:
    print("Attempting to send request...")
    sys.stdout.flush()
    response = requests.post(url, json=data, timeout=10)
    print(f"Request sent. Status code: {response.status_code}")
    sys.stdout.flush()
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: ${result['prediction']:,.2f}")
        print(f"Features used: {result['features_used']}")
    else:
        print(f"Error: {response.json().get('error', 'Unknown error')}")
    
    print(f"Full response: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")

# Testing with missing basic feature
missing_data = {k: v for k, v in data.items() if k != 'grade'}
try:
    response = requests.post(url, json=missing_data)
    print(f"Response status code for missing data: {response.status_code}")
    print(f"Response JSON for missing data: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Request with missing data failed: {e}")

# Getting basic features
features_url = "http://127.0.0.1:5010/v2/basic_features"
try:
    features_response = requests.get(features_url)
    print(f"Basic features: {features_response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Failed to get basic features: {e}")

sys.stdout.flush()
print("All tests completed.")