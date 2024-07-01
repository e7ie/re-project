from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import pandas as pd
import os
import logging
from werkzeug.exceptions import BadRequest
import socket

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model and feature names using absolute paths
model_dir = os.path.join(os.path.dirname(__file__), '../../model/v2/Gradient_Boosting')
with open(os.path.join(model_dir, 'model.pkl'), 'rb') as model_file:
    model = pickle.load(model_file)
with open(os.path.join(model_dir, 'model_features.json'), 'r') as features_file:
    model_features = json.load(features_file)

# Load demographic data using an absolute path
data_dir = os.path.join(os.path.dirname(__file__), '../../data')
demographics_df = pd.read_csv(os.path.join(data_dir, 'zipcode_demographics.csv'), dtype={'zipcode': str})

def prepare_data(input_data):
    """Prepares input data by merging with demographics and aligning feature columns."""
    input_df = pd.DataFrame(input_data, index=[0])
    merged_df = input_df.merge(demographics_df, on='zipcode', how='left').drop(columns='zipcode')
    return merged_df[model_features]

@app.route('/', methods=['GET'])
@app.route('/v2/', methods=['GET'])
def home():
    return "Flask server is running - v2!"

@app.route('/v2/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    try:
        request_data = request.get_json()
        if not request_data:
            raise BadRequest("No input data provided")
        
        prepared_data = prepare_data(request_data)
        prediction = model.predict(prepared_data)
        return jsonify({'prediction': prediction[0]})
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def find_free_port(start_port=5003, max_port=5999):
    for port in range(start_port, max_port + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', port))
            s.close()
            return port
        except OSError:
            continue
    return None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    while True:
        try:
            app.run(debug=True, host='0.0.0.0', port=port)
            break
        except OSError:
            print(f"Port {port} is in use, trying to find a free port...")
            port = find_free_port(port + 1)
            if port is None:
                print("No free ports found. Exiting.")
                exit(1)
            print(f"Found free port: {port}")