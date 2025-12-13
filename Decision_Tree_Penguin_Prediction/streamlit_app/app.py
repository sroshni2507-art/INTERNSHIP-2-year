import streamlit as st
import pickle
import os

# 1️⃣ Ensure the model folder exists
model_folder = "../model"
os.makedirs(model_folder, exist_ok=True)

# 2️⃣ Path to save the model
model_path = os.path.join(model_folder, "penguin_dtree_model.pkl")

# 3️⃣ Save the trained model
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model successfully saved at: {model_path}")

# 4️⃣ Optional: Test the model on first 5 test samples
sample_pred = model.predict(X_test[:5])
species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
print("Sample predictions:", [species_map[i] for i in sample_pred])
