import os
import pickle
import streamlit as st

# ---- SAFE gdown import ----
try:
    import gdown
except ImportError:
    os.system("pip install gdown")
    import gdown

# ---- TensorFlow import (optional) ----
try:
    from tensorflow.keras.models import load_model
    TF_OK = True
except:
    TF_OK = False


st.set_page_config(page_title="ML App", layout="centered")
st.title("🚀 Machine Learning Application")

# -----------------------------------
# 🔴 REPLACE WITH YOUR REAL FILE IDs
# -----------------------------------
PKL_FILE_ID = "https://drive.google.com/file/d/1bMdf8i71eLlVxsT48C2E26OfC79HETyU/view?usp=drive_link"
H5_FILE_ID  = "https://drive.google.com/file/d/1jNa9LLSnKqx53mPtx4EQpvcKOcapaUPM/view?usp=drive_link"

PKL_PATH = "model.pkl"
H5_PATH  = "model.h5"


def download_from_drive(file_id, output):
    if not os.path.exists(output):
        with st.spinner(f"Downloading {output}..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=False, fuzzy=True)


# ---- DOWNLOAD FILES ----
if PKL_FILE_ID != "PUT_REAL_PKL_FILE_ID":
    download_from_drive(PKL_FILE_ID, PKL_PATH)

if H5_FILE_ID != "PUT_REAL_H5_FILE_ID":
    download_from_drive(H5_FILE_ID, H5_PATH)


# ---- LOAD PKL ----
if os.path.exists(PKL_PATH):
    with open(PKL_PATH, "rb") as f:
        pkl_model = pickle.load(f)
    st.success("✅ PKL model loaded")

# ---- LOAD H5 ----
if TF_OK and os.path.exists(H5_PATH):
    h5_model = load_model(H5_PATH)
    st.success("✅ H5 model loaded")

st.info("App ready 🎉")
