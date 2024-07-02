from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import pandas as pd
import numpy as np
import os
import logging

app = Flask(__name__)
CORS(app)  
logging.basicConfig(level=logging.INFO)


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_dir = os.path.join(root_dir, 'model', 'v2', 'Gradient_Boosting')

# Subset of features for basic prediction
BASIC_FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
    'sqft_living15', 'sqft_lot15'
]

try:
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as model_file:
        model = pickle.load(model_file)
    with open(os.path.join(model_dir, 'model_features.json'), 'r') as features_file:
        ALL_FEATURES = json.load(features_file)
except Exception as e:
    logging.error(f"Error loading model or features: {str(e)}")
    raise

def prepare_data(input_data, use_basic_features=False):
    """Prepares input data by aligning feature columns."""
    input_df = pd.DataFrame(input_data, index=[0])
    
    features_to_use = BASIC_FEATURES if use_basic_features else ALL_FEATURES
    
    # DataFrame with all required features, initialized to 0
    prepared_df = pd.DataFrame(0, index=[0], columns=ALL_FEATURES)
    
    for feature in features_to_use:
        if feature in input_df.columns:
            prepared_df[feature] = input_df[feature]
    
    return prepared_df

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
            'features_used': ALL_FEATURES
        })
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/v2/predict_basic', methods=['POST'])
def predict_basic():
    """BONUS: Endpoint for making predictions using only the basic features."""
    try:
        request_data = request.json
        missing_features = set(BASIC_FEATURES) - set(request_data.keys())
        if missing_features:
            return jsonify({'error': f'Missing basic features: {missing_features}'}), 400
        
        prepared_data = prepare_data(request_data, use_basic_features=True)
        prediction = model.predict(prepared_data)
        
        return jsonify({
            'prediction': prediction[0],
            'features_used': BASIC_FEATURES
        })
    except Exception as e:
        logging.error(f"Prediction error with basic features: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/v2/basic_features', methods=['GET'])
def get_basic_features():
    """Endpoint to get the list of basic features."""
    return jsonify({'basic_features': BASIC_FEATURES})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)