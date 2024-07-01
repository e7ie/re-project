# api/v2/main.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import pandas as pd
import os
import logging

app = Flask(__name__)
CORS(app)  
logging.basicConfig(level=logging.INFO)

# Determine the root directory of the project
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_dir = os.path.join(root_dir, 'model', 'v2', 'Gradient_Boosting')
data_dir = os.path.join(root_dir, 'data')

# Define the list of required features
# This list can be easily edited
REQUIRED_FEATURES = [
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'waterfront',
    'view',
    'condition',
    'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built',
    'yr_renovated',
    'zipcode',
    'lat',
    'long',
    'sqft_living15',
    'sqft_lot15'
]

try:
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as model_file:
        model = pickle.load(model_file)
    with open(os.path.join(model_dir, 'model_features.json'), 'r') as features_file:
        model_features = json.load(features_file)
    
    # Load demographic data
    demographics_df = pd.read_csv(os.path.join(data_dir, 'zipcode_demographics.csv'), dtype={'zipcode': str})
except Exception as e:
    logging.error(f"Error loading model or data: {str(e)}")
    raise

def prepare_data(input_data, use_required_features=False):
    """Prepares input data by merging with demographics and aligning feature columns."""
    input_df = pd.DataFrame(input_data, index=[0])
    merged_df = input_df.merge(demographics_df, on='zipcode', how='left').drop(columns='zipcode')
    features_to_use = REQUIRED_FEATURES if use_required_features else model_features
    return merged_df[features_to_use]

@app.route('/', methods=['GET'])
@app.route('/v2/', methods=['GET'])
def home():
    return "Flask server is running - v2. Let's predict house prices!"

@app.route('/v2/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions using all features."""
    try:
        request_data = request.json
        prepared_data = prepare_data(request_data)
        prediction = model.predict(prepared_data)
        return jsonify({
            'prediction': prediction[0],
            'features_used': list(prepared_data.columns)
        })
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/v2/predict_required', methods=['POST'])
def predict_required():
    """BONUS: Endpoint for making predictions using only the required features."""
    try:
        request_data = request.json
        prepared_data = prepare_data(request_data, use_required_features=True)
        prediction = model.predict(prepared_data)
        return jsonify({
            'prediction': prediction[0],
            'features_used': REQUIRED_FEATURES
        })
    except Exception as e:
        logging.error(f"Prediction error with required features: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/v2/required_features', methods=['GET'])
def get_required_features():
    """Endpoint to get the list of required features."""
    return jsonify({'required_features': REQUIRED_FEATURES})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)