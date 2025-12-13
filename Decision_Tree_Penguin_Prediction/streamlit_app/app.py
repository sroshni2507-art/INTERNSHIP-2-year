import streamlit as st
import pickle
import numpy as np
import os

# In your app.py file:

# Construct the file path relative to the repository root.
# This format is reliable on Streamlit Cloud (Linux environment).
MODEL_PATH = "Decision_Tree_Penguin_Prediction/model/penguin_dtree_model (2).pkl"

# 2️⃣ Load model safely
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
else:
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except EOFError:
        st.error("Error loading the model. The file may be corrupted.")
    else:
        # 3️⃣ App title
        st.title("Penguin Species Prediction (Decision Tree)")

        # 4️⃣ Input fields
        bill_length = st.number_input("Bill Length (mm)", 0.0)
        bill_depth = st.number_input("Bill Depth (mm)", 0.0)
        flipper_length = st.number_input("Flipper Length (mm)", 0.0)
        body_mass = st.number_input("Body Mass (g)", 0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        island = st.selectbox("Island", ["Biscoe", "Dream", "Torgersen"])

        # 5️⃣ Encode categorical inputs
        sex_val = 1 if sex == "Male" else 0
        island_map = {"Biscoe": 0, "Dream": 1, "Torgersen": 2}
        island_val = island_map[island]

        # 6️⃣ Predict button
        if st.button("Predict"):
            data = np.array([[bill_length, bill_depth, flipper_length, body_mass, sex_val, island_val]])
            pred = model.predict(data)
            species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
            st.success(f"Predicted Species: {species_map[pred[0]]}")
