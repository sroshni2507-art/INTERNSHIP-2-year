import streamlit as st
import pickle
import numpy as np
import os

# Load model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../model/telco_churn_model.pkl"
)

model = pickle.load(open(MODEL_PATH, "rb"))

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
