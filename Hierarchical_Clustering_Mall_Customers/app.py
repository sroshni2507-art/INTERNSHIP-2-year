import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

st.title("🛍️ Mall Customer Segmentation")
st.write("Hierarchical Clustering (Unsupervised Learning)")

# Load model
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "hierarchical_mall_customer.pkl")

model = joblib.load(MODEL_PATH)

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Mall Customers CSV file", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Drop unwanted columns
    df_clean = df.drop(
        columns=["CustomerID", "Genre", "Gender"],
        errors="ignore"
    )

    # Use only numeric columns
    X = df_clean.select_dtypes(include=["int64", "float64"])

    # Apply hierarchical clustering
    clusters = model.fit_predict(X)
    df_clean["Cluster"] = clusters

    st.subheader("📊 Clustered Data")
    st.dataframe(df_clean.head())

    # Visualization
    st.subheader("📈 Cluster Visualization")

    fig, ax = plt.subplots()
    ax.scatter(
        df_clean["Annual Income (k$)"],
        df_clean["Spending Score (1-100)"],
        c=df_clean["Cluster"]
    )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    st.pyplot(fig)

    # ---------------- OPTIONAL USER INPUT ----------------
    st.subheader("🔍 Find Cluster for a New Customer")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input(
        "Annual Income (k$)", min_value=10, max_value=200, value=50
    )
    score = st.number_input(
        "Spending Score (1-100)", min_value=1, max_value=100, value=50
    )

    if st.button("Find Cluster"):
        new_customer = pd.DataFrame(
            [[age, income, score]],
            columns=[
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        )

        temp_data = pd.concat([X, new_customer], ignore_index=True)
        temp_clusters = model.fit_predict(temp_data)

        st.success(
            f"🟢 This customer belongs to Cluster: {temp_clusters[-1]}"
        )

else:
    st.info("Please upload the Mall Customers CSV file.")
