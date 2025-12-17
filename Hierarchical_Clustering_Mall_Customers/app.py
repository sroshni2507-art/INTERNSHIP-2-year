import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mall Customer Segmentation")

st.title("🛍️ Mall Customer Segmentation")
st.write("Hierarchical Clustering – Customer Groups")

# Load model
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "hierarchical_mall_customer.pkl")
model = joblib.load(MODEL_PATH)

# Cluster meaning
cluster_names = {
    0: "High Income – High Spending Customers 💎",
    1: "Low Income – Low Spending Customers 🪙",
    2: "High Income – Low Spending Customers 💼",
    3: "Low Income – High Spending Customers 🎯"
}

# Upload CSV
uploaded_file = st.file_uploader("Upload Mall Customers CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Preprocessing
    df_clean = df.drop(
        columns=["CustomerID", "Genre", "Gender"],
        errors="ignore"
    )

    X = df_clean.select_dtypes(include=["int64", "float64"])

    # Clustering
    clusters = model.fit_predict(X)
    df_clean["Cluster"] = clusters
    df_clean["Customer Type"] = df_clean["Cluster"].map(cluster_names)

    st.subheader("📊 Clustered Customers")
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

    # -------- USER INPUT ----------
    st.subheader("🔍 Customer Segmentation (Manual Input)")

    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual Income (k$)", 10, 200, 50)
    score = st.number_input("Spending Score (1-100)", 1, 100, 50)

    if st.button("Find Customer Type"):
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

        cluster_id = temp_clusters[-1]
        customer_type = cluster_names.get(
            cluster_id, "Unknown Customer Group"
        )

        st.success(f"🟢 Customer Type: {customer_type}")

else:
    st.info("Please upload the Mall Customers CSV file.")
