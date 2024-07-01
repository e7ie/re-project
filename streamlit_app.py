# streamlit_app.py
import streamlit as st
import requests
import json

st.title('Real Estate Price Prediction')

# Input fields
bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, value=3)
bathrooms = st.number_input('Bathrooms', min_value=0, max_value=10, value=2)
sqft_living = st.number_input('Square Feet Living', min_value=0, value=1800)
sqft_lot = st.number_input('Square Feet Lot', min_value=0, value=5000)
floors = st.number_input('Floors', min_value=0.0, max_value=5.0, value=2.0)
waterfront = st.selectbox('Waterfront', [0, 1])
view = st.selectbox('View', [0, 1, 2, 3, 4])
condition = st.selectbox('Condition', [1, 2, 3, 4, 5])
grade = st.selectbox('Grade', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
sqft_above = st.number_input('Square Feet Above', min_value=0, value=1500)
sqft_basement = st.number_input('Square Feet Basement', min_value=0, value=300)
yr_built = st.number_input('Year Built', min_value=1900, max_value=2021, value=2000)
yr_renovated = st.number_input('Year Renovated', min_value=1900, max_value=2021, value=2000)
zipcode = st.text_input('Zipcode', '98103')
lat = st.number_input('Latitude', value=47.5112)
long = st.number_input('Longitude', value=-122.257)
sqft_living15 = st.number_input('Square Feet Living 15', min_value=0, value=1800)
sqft_lot15 = st.number_input('Square Feet Lot 15', min_value=0, value=5000)

data = {
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "waterfront": waterfront,
    "view": view,
    "condition": condition,
    "grade": grade,
    "sqft_above": sqft_above,
    "sqft_basement": sqft_basement,
    "yr_built": yr_built,
    "yr_renovated": yr_renovated,
    "zipcode": zipcode,
    "lat": lat,
    "long": long,
    "sqft_living15": sqft_living15,
    "sqft_lot15": sqft_lot15
}

api_version = st.selectbox('API Version', ['v1', 'v2'])

# Add option for required features only if v2 is selected
use_required_features = False
if api_version == 'v2':
    use_required_features = st.checkbox('Use only required features')

# Button to make prediction
if st.button('Predict'):
    endpoint = f'/{api_version}/{"predict_required" if use_required_features else "predict"}'
    response = requests.post(f'http://flask_api_{api_version}:5003{endpoint}', json=data)
    if response.status_code == 200:
        result = response.json()
        prediction = result['prediction']
        st.success(f'The predicted price is ${prediction:,.2f}')
        if 'features_used' in result:
            st.info(f'Features used: {", ".join(result["features_used"])}')
    else:
        st.error('Error: Could not get prediction')