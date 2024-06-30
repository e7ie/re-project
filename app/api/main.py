from flask import Flask, request, jsonify  # Import necessary modules from Flask
import pickle  # Import the pickle module for loading the serialized model
import json  # Import the JSON module for loading feature names
import pandas as pd  # Import pandas for data manipulation
import os

app = Flask(__name__)  # Initialize the Flask app

# Determine the root directory of the project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load the model and feature names using absolute paths
with open(os.path.join(root_dir, 'model', 'model.pkl'), 'rb') as model_file:
    model = pickle.load(model_file)
with open(os.path.join(root_dir, 'model', 'model_features.json'), 'r') as features_file:
    model_features = json.load(features_file)

# Load demographic data using an absolute path
demographics_df = pd.read_csv(os.path.join(root_dir, 'data', 'zipcode_demographics.csv'), dtype={'zipcode': str})

def prepare_data(input_data):
    """Prepares input data by merging with demographics and aligning feature columns."""
    input_df = pd.DataFrame(input_data, index=[0])
    merged_df = input_df.merge(demographics_df, on='zipcode', how='left').drop(columns='zipcode')
    return merged_df[model_features]

@app.route('/', methods=['GET'])
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    request_data = request.json  # Parse JSON data from the request
    prepared_data = prepare_data(request_data)  # Prepare the input data for prediction
    prediction = model.predict(prepared_data)  # Make the prediction
    return jsonify({'prediction': prediction[0]})  # Return the prediction as JSON

if __name__ == '__main__':
    app.run(debug=True, port=5003)
