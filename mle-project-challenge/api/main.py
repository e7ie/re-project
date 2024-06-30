from flask import Flask, request, jsonify  # Import necessary modules from Flask
import pickle  # Import the pickle module for loading the serialized model
import json  # Import the JSON module for loading feature names
import pandas as pd  # Import pandas for data manipulation

app = Flask(__name__)  # Initialize the Flask app

# Load the model and feature names
with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('model/model_features.json', 'r') as features_file:
    model_features = json.load(features_file)

# Load demographic data
demographics_df = pd.read_csv('data/zipcode_demographics.csv', dtype={'zipcode': str})

def prepare_data(input_data):
    """Prepares input data by merging with demographics and aligning feature columns."""
    print(f"Received input data: {input_data}")  # Debug statement
    input_df = pd.DataFrame(input_data, index=[0])
    merged_df = input_df.merge(demographics_df, on='zipcode', how='left').drop(columns='zipcode')
    print(f"Prepared data for prediction: {merged_df[model_features]}")  # Debug statement
    return merged_df[model_features]

@app.route('/', methods=['GET'])
def home():
    print("Root route accessed")  # Debug statement
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    print("Predict route accessed")  # Debug statement
    request_data = request.json  # Parse JSON data from the request
    print(f"Request data: {request_data}")  # Debug statement
    prepared_data = prepare_data(request_data)  # Prepare the input data for prediction
    prediction = model.predict(prepared_data)  # Make the prediction
    print(f"Prediction: {prediction[0]}")  # Debug statement
    return jsonify({'prediction': prediction[0]})  # Return the prediction as JSON

if __name__ == '__main__':
    app.run(debug=True, port=5003)  # Run the app on port 5002
