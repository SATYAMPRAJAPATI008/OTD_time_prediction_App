import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import urllib.request


# Google Drive file ID (Replace with your actual file ID)
GDRIVE_FILE_ID = "1DvfGR9pJqobmIolgTYWk7EXS3K3Z7qIS"

# Construct the direct download URL
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

@st.cache_resource
def load_model():
    urllib.request.urlretrieve(MODEL_URL, "voting_model.pkl")
    return pickle.load(open("voting_model.pkl", "rb"))

# Load the model
voting_model = load_model()

# Set the Streamlit page configuration
st.set_page_config(page_title="OTD Time Forecasting", page_icon=":truck:", layout="wide")


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
