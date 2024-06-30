import streamlit as st
import requests

st.title('Real Estate Price Prediction')

# Input fields
bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, value=3)
bathrooms = st.number_input('Bathrooms', min_value=0, max_value=10, value=2)
sqft_living = st.number_input('Square Feet Living', min_value=0, value=1800)
sqft_lot = st.number_input('Square Feet Lot', min_value=0, value=5000)
floors = st.number_input('Floors', min_value=0.0, max_value=5.0, value=2.0)
sqft_above = st.number_input('Square Feet Above', min_value=0, value=1500)
sqft_basement = st.number_input('Square Feet Basement', min_value=0, value=300)
zipcode = st.text_input('Zipcode', '98103')

data = {
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "sqft_above": sqft_above,
    "sqft_basement": sqft_basement,
    "zipcode": zipcode
}

# Button to make prediction
if st.button('Predict'):
    response = requests.post('http://127.0.0.1:5003/predict', json=data)
    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.success(f'The predicted price is ${prediction:,.2f}')
    else:
        st.error('Error: Could not get prediction')
