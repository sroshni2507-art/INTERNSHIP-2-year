import os
import pickle
import streamlit as st

# ---------- SAFE gdown import ----------
try:
    import gdown
except ImportError:
    os.system("pip install gdown")
    import gdown

# ---------- OPTIONAL: TensorFlow import ----------
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False


st.set_page_config(page_title="ML App", layout="centered")
st.title("🚀 Machine Learning Application")

# ---------- GOOGLE DRIVE FILE IDs ----------
# ⚠️ Folder ID use pannakoodathu
# Each file-ku individual FILE ID venum

PKL_FILE_ID = "https://drive.google.com/file/d/1bMdf8i71eLlVxsT48C2E26OfC79HETyU/view?usp=drive_link"
H5_FILE_ID  = "https://drive.google.com/file/d/1jNa9LLSnKqx53mPtx4EQpvcKOcapaUPM/view?usp=drive_link"

# ---------- LOCAL PATHS ----------
PKL_PATH = "model.pkl"
H5_PATH  = "model.h5"


# ---------- DOWNLOAD FUNCTIONS ----------
def download_file(file_id, output_path):
    if not os.path.exists(output_path):
        with st.spinner(f"Downloading {output_path} from Google Drive..."):
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                output_path,
                quiet=False
            )


# ---------- DOWNLOAD FILES ----------
if PKL_FILE_ID != "PUT_YOUR_PKL_FILE_ID_HERE":
    download_file(PKL_FILE_ID, PKL_PATH)

if H5_FILE_ID != "PUT_YOUR_H5_FILE_ID_HERE":
    download_file(H5_FILE_ID, H5_PATH)


# ---------- LOAD PKL MODEL ----------
pkl_model = None
if os.path.exists(PKL_PATH):
    with open(PKL_PATH, "rb") as f:
        pkl_model = pickle.load(f)
    st.success("✅ PKL model loaded successfully")


# ---------- LOAD H5 MODEL ----------
h5_model = None
if TENSORFLOW_AVAILABLE and os.path.exists(H5_PATH):
    h5_model = load_model(H5_PATH)
    st.success("✅ H5 model loaded successfully")
elif not TENSORFLOW_AVAILABLE:
    st.warning("⚠️ TensorFlow not installed. H5 model skipped.")


# ---------- SIMPLE TEST UI ----------
st.subheader("🔍 Test Section")

if pkl_model or h5_model:
    st.write("Models are ready to use 🎉")
else:
    st.error("❌ Models not loaded. Check Drive File IDs.")
