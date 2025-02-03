import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.sidebar.title("Predictive Maintenance Dashboard")
st.sidebar.info("Upload data and predict machine failures.")

uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())
    
    if st.button("Predict Failure"):
        data_json = {"data": data.values.tolist(), "columns":  data.columns.tolist()}
        response = requests.post("http://127.0.0.1:5000/predict", json=data_json)
        
        if response.status_code == 200:
            prediction = response.json()
            st.subheader("Prediction Results")
            if "prediction" in prediction:
                st.write(prediction["prediction"])
            elif "error" in prediction:
                st.error(f"Error in prediction: {prediction['error']}")
            else:
                st.error("Unexpected response format.")
        else:
            st.error(f"Error in making predictions: {response.status_code} - {response.text}")