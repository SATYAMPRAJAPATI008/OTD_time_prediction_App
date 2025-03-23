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

# Set page config (title and icon)
st.set_page_config(page_title="Delivery Time Estimator", page_icon="🚚", layout="centered")

# App header
st.title("🚚 Order Delivery Time Estimator")
st.markdown("Use this app to estimate the expected delivery time for your order based on input details.")

st.divider()

# Create two columns for inputs
col1, col2 = st.columns(2)

# Input fields in the first column
with col1:
    purchase_dow = st.selectbox("Select Day of the Week", 
                                options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                index=3)
    purchase_dow = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(purchase_dow)
    
    purchase_month = st.selectbox("Select Month", 
                                  options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 
                                  index=0)
    purchase_month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"].index(purchase_month) + 1
    
    year = st.number_input("Enter Year", min_value=2000, max_value=2030, value=2023, step=1)

# Input field in the second column
with col2:
    distance = st.slider("Distance to Destination (km)", min_value=1.0, max_value=1000.0, value=475.35, step=0.1)
    
st.divider()

# Display the prediction
if st.button("Estimate Delivery Time"):
    features = [purchase_dow, purchase_month, year, distance]
    prediction = predict_delivery_time(features)
    st.success(f"🎉 Estimated Delivery Time: **{prediction} days**")

# Footer
st.markdown("---")
st.markdown("<small>Developed with ❤️ by a robotics and automation engineer.</small>", unsafe_allow_html=True)
