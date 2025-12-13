import streamlit as st

import numpy as np

import os
import pickle

# Get the directory of the current script (app.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Join the script directory with the model file name
model_path = os.path.join(script_dir, "telco_churn_model.pkl")

# Load the model using the constructed path
with open(model_path, "rb") as file:
    model = pickle.load(file)


st.title("Telco Customer Churn Prediction")

st.write("Enter customer details:")

monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
tenure = st.number_input("Tenure (Months)", min_value=0)
senior_citizen = st.selectbox("Senior Citizen", [0, 1])

if st.button("Predict"):
    input_data = np.array([[monthly_charges, tenure, senior_citizen]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer is likely to CHURN ❌")
    else:
        st.success("Customer will NOT churn ✅")
