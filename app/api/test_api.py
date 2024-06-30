import requests

# URL of the Flask API
url = "http://127.0.0.1:5003/predict"

# JSON data to be sent in the POST request
data = {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 2,
    "sqft_above": 1500,
    "sqft_basement": 300,
    "zipcode": "98103"
}

# Sending the POST request with JSON data
try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise an exception for HTTP errors
    print(f"Response status code: {response.status_code}")  # Debug statement
    print(f"Response JSON: {response.json()}")  # Print the server's response
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
