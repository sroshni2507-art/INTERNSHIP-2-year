import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("🛍️ Mall Customer Segmentation")
st.write("Hierarchical Clustering (Unsupervised Learning)")

# Step 1: Upload the model and scaler (avoids FileNotFoundError)
uploaded_model = st.file_uploader("Upload Hierarchical Clustering Model (.pkl)", type=["pkl"])
uploaded_scaler = st.file_uploader("Upload Scaler (.pkl)", type=["pkl"])

if uploaded_model and uploaded_scaler:
    model = joblib.load(uploaded_model)
    scaler = joblib.load(uploaded_scaler)

    # Step 2: Upload Customer CSV
    uploaded_csv = st.file_uploader("Upload Mall Customer CSV", type=["csv"])

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Step 3: Preprocessing
        if "CustomerID" in df.columns:
            df = df.drop("CustomerID", axis=1)

        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

        # Step 4: Scale features
        X_scaled = scaler.transform(df)

        # Step 5: Predict clusters
        clusters = model.fit_predict(X_scaled)
        df["Cluster"] = clusters

        st.subheader("Clustered Data")
        st.dataframe(df.head())

        # Step 6: Visualization
        st.subheader("Cluster Visualization")
        plt.figure(figsize=(6, 4))
        plt.scatter(
            df["Annual Income (k$)"],
            df["Spending Score (1-100)"],
            c=df["Cluster"],
            cmap="viridis"
        )
        plt.xlabel("Annual Income")
        plt.ylabel("Spending Score")
        plt.title("Customer Segmentation Clusters")
        st.pyplot(plt)
else:
    st.warning("Please upload both the model (.pkl) and scaler (.pkl) files to proceed.")
