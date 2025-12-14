import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification App")
st.write("Predict the **Iris flower species** using a **Decision Tree model**.")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = sns.load_dataset("iris")
    return df

df = load_data()

# ==============================
# MODEL TRAINING
# ==============================
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.header("🔢 Input Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)", 
    float(X.sepal_length.min()), 
    float(X.sepal_length.max()), 
    float(X.sepal_length.mean())
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)", 
    float(X.sepal_width.min()), 
    float(X.sepal_width.max()), 
    float(X.sepal_width.mean())
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)", 
    float(X.petal_length.min()), 
    float(X.petal_length.max()), 
    float(X.petal_length.mean())
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)", 
    float(X.petal_width.min()), 
    float(X.petal_width.max()), 
    float(X.petal_width.mean())
)

# ==============================
# PREDICTION
# ==============================
input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=X.columns
)

prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# ==============================
# OUTPUT
# ==============================
st.subheader("🔍 Prediction Result")

st.success(f"🌼 Predicted Species: **{prediction[0]}**")

st.subheader("📊 Prediction Probability")
st.bar_chart(
    pd.DataFrame(
        prediction_proba,
        columns=model.classes_
    ).T
)

# ==============================
# SHOW DATASET
# ==============================
with st.expander("📁 View Iris Dataset"):
    st.dataframe(df)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("**Machine Learning Project | Decision Tree | Iris Dataset**")
