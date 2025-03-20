import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
@st.cache_resource
def load_model():
    return joblib.load("voting_model_compressed.pkl")

model_data = load_model()
voting_model = model_data['model']
scaler = model_data['scaler']

# Function for prediction
def predict_delivery_time(features):
    features_scaled = scaler.transform([features])
    return round(voting_model.predict(features_scaled)[0])

# Streamlit UI
st.title("Order Delivery Time Estimator")

purchase_dow = st.slider("Day of the Week", 0, 6, 3)
purchase_month = st.slider("Month", 1, 12, 1)
year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)
product_size_cm3 = st.number_input("Product Size (cm³)", min_value=100, value=9328)
product_weight_g = st.number_input("Product Weight (g)", min_value=100, value=1800)
geolocation_state_customer = st.number_input("Customer Location Code", min_value=1, value=10)
geolocation_state_seller = st.number_input("Seller Location Code", min_value=1, value=20)
distance = st.number_input("Distance (km)", min_value=1.0, value=475.35)

if st.button("Predict Delivery Time"):
    features = ['purchase_dow', 'purchase_month', 'year', 'distance']

    prediction = predict_delivery_time(features)
    st.success(f"Estimated Delivery Time: {prediction} days")
