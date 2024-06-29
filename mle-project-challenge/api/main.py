from flask import Flask, request, jsonify  # Import necessary modules from Flask
import pickle  # Import the pickle module for loading the model
import json  # Import the JSON module for loading feature names
import pandas as pd  # Import pandas for data manipulation

app = Flask(__name__)  # Create a Flask application instance

# Load model and features
model = pickle.load(open('app/model/model.pkl', 'rb'))  # Load the trained model
with open('app/model/model_features.json', 'r') as f:  # Load the feature names
    model_features = json.load(f)

# Load demographic data
demographics = pd.read_csv('data/zipcode_demographics.csv', dtype={'zipcode': str})  # Load the demographics data

def prepare_input(data):
    """Merge input data with demographics and reorder columns."""
    input_df = pd.DataFrame(data, index=[0])  # Convert input data to DataFrame
    merged_df = input_df.merge(demographics, on="zipcode", how="left").drop(columns="zipcode")  # Merge with demographics
    return merged_df[model_features]  # Reorder columns to match model features

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Parse JSON data from the request
    input_data = prepare_input(data)  # Prepare input data
    prediction = model.predict(input_data)  # Make a prediction
    return jsonify({'prediction': prediction[0]})  # Return the prediction as JSON

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
