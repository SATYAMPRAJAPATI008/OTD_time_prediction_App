import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import urllib.request
import os
import requests

MODEL_RF_SVR_URL = "https://github.com/Housefly-hub/OTD_Time_Forecasting_App/releases/download/v1.0/rf_svr_voting.pkl"
MODEL_XGB_URL = "https://github.com/Housefly-hub/OTD_Time_Forecasting_App/releases/download/v1.0/xgboost_model.json"

# Set the Streamlit page configuration
st.set_page_config(page_title="OTD Time Forecasting", page_icon=":truck:", layout="wide")
# Function to check if the file is correctly downloaded
def download_file(url, filename):
    """Download a file from GitHub Releases."""
    try:
        print(f"Downloading {filename} from GitHub Releases...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):  # 1MB chunks
                f.write(chunk)

        print(f"✅ Download complete: {filename}")

    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        raise



@st.cache_resource
def load_models():
    """Load both the Voting Regressor and XGBoost models."""
    download_file(MODEL_RF_SVR_URL, "rf_svr_voting.pkl")
    download_file(MODEL_XGB_URL, "xgboost_model.json")

    # Load XGBoost properly
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgboost_model.json")

    # Load the Voting Regressor
    with open("rf_svr_voting.pkl", "rb") as f:
        voting_model = pickle.load(f)

    return voting_model, xgb_model

# Load models
voting_model, xgb_model = load_models()




# Function for predicting wait time
def wait_time_predictor(purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
                        geolocation_state_customer, geolocation_state_seller, distance):
    prediction = voting_model.predict(
        np.array([[purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
                   geolocation_state_customer, geolocation_state_seller, distance]])
    )
    return round(prediction[0])

# UI Design
st.title("OTD Time Forecasting: Predict Order to Delivery Time")
st.caption("This tool predicts the estimated delivery time using an ensemble ML model trained on order data.")

with st.sidebar:
    st.header("Input Order Details")
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Customer Location Code", value=10)
    geolocation_state_seller = st.number_input("Seller Location Code", value=20)
    distance = st.number_input("Distance (km)", value=475.35)
    submit = st.button("Predict Delivery Time")

with st.container():
    st.header("Predicted Delivery Time (in Days)")
    if submit:
        prediction = wait_time_predictor(purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
                                         geolocation_state_customer, geolocation_state_seller, distance)
        st.success(f"Estimated Wait Time: {prediction} days")

    # Display a sample dataset
    data = {
        "Day of the Week": [0, 3, 1],
        "Month": [6, 3, 1],
        "Year": [2018, 2017, 2018],
        "Product Size (cm³)": [37206, 63714, 54816],
        "Product Weight (g)": [16250, 7249, 9600],
        "Customer Location": [25, 25, 25],
        "Seller Location": [20, 7, 20],
        "Distance (km)": [247.94, 250.35, 4.915]
    }
    df = pd.DataFrame(data)
    st.header("Sample Data")
    st.write(df)
