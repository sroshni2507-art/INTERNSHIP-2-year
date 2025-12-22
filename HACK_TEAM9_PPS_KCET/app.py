import os
import gdown
import pickle
from tensorflow.keras.models import load_model

# -------------------------------
# Google Drive FILE IDs
# -------------------------------
PKL_FILE_ID = "https://drive.google.com/file/d/1bMdf8i71eLlVxsT48C2E26OfC79HETyU/view?usp=drive_link"
H5_FILE_ID = "https://drive.google.com/file/d/1jNa9LLSnKqx53mPtx4EQpvcKOcapaUPM/view?usp=drive_link"

# Local paths
PKL_PATH = "model.pkl"
H5_PATH = "model.h5"

# -------------------------------
# Download PKL if not exists
# -------------------------------
if not os.path.exists(PKL_PATH):
    print("Downloading PKL model...")
    gdown.download(
        f"https://drive.google.com/uc?id={PKL_FILE_ID}",
        PKL_PATH,
        quiet=False
    )

# -------------------------------
# Download H5 if not exists
# -------------------------------
if not os.path.exists(H5_PATH):
    print("Downloading H5 model...")
    gdown.download(
        f"https://drive.google.com/uc?id={H5_FILE_ID}",
        H5_PATH,
        quiet=False
    )

# -------------------------------
# Load models
# -------------------------------
with open(PKL_PATH, "rb") as f:
    pkl_model = pickle.load(f)

h5_model = load_model(H5_PATH)

print("✅ Models loaded successfully")

