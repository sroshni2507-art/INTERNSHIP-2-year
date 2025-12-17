import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

st.title("🛍️ Mall Customer Segmentation")
st.write("Hierarchical Clustering using saved .pkl model")

# -----------------------------
# Load model & scaler safely
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "hierarchical_mall_customer.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Mall Customers CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # PREPROCESSING (IMPORTANT FIX)
    # -----------------------------
    # Drop unused columns safely
    df = df.drop(columns=["CustomerID", "Genre", "Gender"], errors="ignore")

    # Use EXACT features used during training
    required_cols = [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    # Check columns
    if not all(col in df.columns for col in required_cols):
        st.error("CSV file does not contain required columns!")
    else:
        X = df[required_cols]

        # Scale data
        X_scaled = scaler.transform(X)

        # Hierarchical clustering (fit again – unsupervised)
        clusters = model.fit_predict(X_scaled)

        df["Cluster"] = clusters

        st.subheader("🧩 Clustered Data")
        st.dataframe(df)

        # -----------------------------
        # Visualization
        # -----------------------------
        st.subheader("📈 Cluster Visualization")

        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(
            df["Annual Income (k$)"],
            df["Spending Score (1-100)"],
            c=df["Cluster"]
        )
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.set_title("Mall Customer Segments")

        st.pyplot(fig)

else:
    st.info("⬆️ Please upload the Mall Customers CSV file.")
