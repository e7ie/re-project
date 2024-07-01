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
model_dir = os.path.join(root_dir, 'model', 'v1')
data_dir = os.path.join(root_dir, 'data')

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

def prepare_data(input_data):
    """Prepares input data by merging with demographics and aligning feature columns."""
    input_df = pd.DataFrame(input_data, index=[0])
    merged_df = input_df.merge(demographics_df, on='zipcode', how='left').drop(columns='zipcode')
    return merged_df[model_features]

@app.route('/', methods=['GET'])
def home():
    return "Flask server is running. Let's predict house prices!"

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    try:
        request_data = request.json
        prepared_data = prepare_data(request_data)
        prediction = model.predict(prepared_data)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)