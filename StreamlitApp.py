import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
@st.cache_resource
def load_model():
    return joblib.load("voting_model_compressed.pkl")

model_data = load_model()
voting_model = model_data['model']
scaler = model_data['scaler']

# Custom Futuristic CSS Styling
st.markdown(
    """
    <style>
        body {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stApp {
            background: linear-gradient(135deg, #1f1f1f, #0e1117);
        }
        .css-18e3th9 {
            background-color: #121212;
            border-radius: 10px;
            padding: 20px;
        }
        .stButton>button {
            background-color: #00d4ff;
            color: black;
            font-weight: bold;
            border-radius: 5px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #ff00ff;
            color: white;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #1a1a2e;
            color: #00d4ff;
            border: 1px solid #00d4ff;
            border-radius: 5px;
        }
        .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
            border-color: #ff00ff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚀 Futuristic Order Delivery Time Estimator")
st.markdown("### Enter Order Details Below")

# Input Fields
col1, col2 = st.columns(2)
with col1:
    purchase_dow = st.number_input("📅 Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("📆 Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("🕒 Year", min_value=2000, max_value=2030, value=2023)
with col2:
    distance = st.number_input("📏 Distance (km)", min_value=1.0, value=475.35)

# Prediction Button
if st.button("🔮 Predict Delivery Time"):
    features = [purchase_dow, purchase_month, year, distance]
    features_scaled = scaler.transform([features])
    prediction = round(voting_model.predict(features_scaled)[0])
    st.success(f"🚀 Estimated Delivery Time: {prediction} days")

# Display Cleaned Data Sample
st.markdown("### Sample Order Data")
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_orders.csv").head(10)
st.dataframe(load_data(), height=200, use_container_width=True)