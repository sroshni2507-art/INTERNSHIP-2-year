import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.title("🛍️ Mall Customer Segmentation")
st.write("Hierarchical Clustering using Saved .pkl Model")

# Load saved model
model = joblib.load("hierarchical_mall_customer.pkl")

# Upload CSV
uploaded_file = st.file_uploader("Upload Mall Customers CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Preprocessing
    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    if "Gender" in df.columns:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"])

    # Apply clustering (re-fit is required)
    clusters = model.fit_predict(df)
    df["Cluster"] = clusters

    st.subheader("Clustered Data")
    st.dataframe(df)

    # Visualization
    st.subheader("Cluster Visualization")
    plt.figure(figsize=(6, 4))
    plt.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score")
    st.pyplot(plt)

else:
    st.info("Please upload the Mall Customers CSV file.")
