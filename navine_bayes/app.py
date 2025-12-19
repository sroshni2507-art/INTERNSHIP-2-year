import streamlit as st
import pickle
import pandas as pd
import os  # ✅ Import os for file path handling

# Get the path of the current script
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "nb_model.pkl")  # Full path to your model

# Load your Naive Bayes model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Naive Bayes Predictor", layout="centered")
st.title("Naive Bayes Predictor 🧠")

# Example: 2 input features (change based on your model)
feature1 = st.text_input("Enter Feature 1:")
feature2 = st.text_input("Enter Feature 2:")

if st.button("Predict"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([[feature1, feature2]], columns=["Feature1", "Feature2"])
    
    # Predict
    prediction = model.predict(input_df)
    st.success(f"Prediction: {prediction[0]}")
