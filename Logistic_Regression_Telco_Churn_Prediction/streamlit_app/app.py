import streamlit as st
import pickle
import numpy as np
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../model/telco_churn_model.pkl"
)

model = pickle.load(open(MODEL_PATH, "rb"))

st.title("Telco Customer Churn Prediction")

tenure = st.number_input("Tenure (months)", 0, 100)
monthly = st.number_input("Monthly Charges", 0.0)
total = st.number_input("Total Charges", 0.0)

if st.button("Predict"):
    data = np.array([[tenure, monthly, total]])
    result = model.predict(data)

    if result[0] == 1:
        st.error("Customer is likely to CHURN ❌")
    else:
        st.success("Customer will NOT churn ✅")
