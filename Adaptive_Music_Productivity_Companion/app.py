import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os


BASE_DIR = os.path.dirname(__file__)

nb_task = pickle.load(open(os.path.join(BASE_DIR, "nb_task.pkl"), "rb"))
encoders = pickle.load(open(os.path.join(BASE_DIR, "encoders.pkl"), "rb"))
apriori_rules = pickle.load(open(os.path.join(BASE_DIR, "apriori_rules.pkl"), "rb"))


# Streamlit app
st.set_page_config(page_title="Adaptive Music & Productivity Companion", layout="wide")

st.title("🎵 Adaptive Music & Productivity Companion")
st.markdown("Get personalized **Task** and **Music recommendations** based on your Mood, Activity, Time, and Goal!")

# --- User Inputs ---
col1, col2 = st.columns(2)

with col1:
    mood = st.selectbox("Select Your Mood:", encoders['le_mood'].classes_)
    activity = st.selectbox("Select Your Activity:", encoders['le_activity'].classes_)
    time_of_day = st.slider("Select Time of Day (Hour 0-23):", 0, 23, 14)

with col2:
    goal = st.selectbox("Select Your Goal:", encoders['le_goal'].classes_)

# --- Predict ---
if st.button("Get Recommendation"):
    
    # Rule-based music recommendation
    if mood == "Sad" and activity == "Relaxing":
        music_pred = "Calm Acoustic / Ambient"
    elif mood == "Energetic" and activity == "Workout":
        music_pred = "High BPM EDM / Hip-Hop"
    elif mood == "Calm" and activity == "Studying":
        music_pred = "Lo-fi / Instrumental"
    elif mood == "Stressed" and activity == "Coding":
        music_pred = "Low-lyric Electronic"
    else:
        music_pred = "Soft Background Music"

    st.success(f"🎧 Recommended Music Type: {music_pred}")

    # Encode inputs
    X_user = np.array([[encoders['le_mood'].transform([mood])[0],
                        encoders['le_activity'].transform([activity])[0],
                        time_of_day,
                        encoders['le_goal'].transform([goal])[0]]])

    # Task Prediction
    task_pred = encoders['le_task'].inverse_transform(nb_task.predict(X_user))[0]

    # Music Prediction
    music_pred = encoders['le_music'].inverse_transform(knn_music.predict(X_user))[0]

    st.subheader("✅ Recommendations")
    st.write(f"**Recommended Task:** {task_pred}")
    st.write(f"**Recommended Music:** {music_pred}")

    # Optional: Show example playlist link
    # You can add this to your dataset or hardcode a sample
    playlist_links = {
        "Lo-Fi": "https://www.youtube.com/watch?v=3AtDnEC4zak",
        "Electronic": "https://www.youtube.com/watch?v=HMnrl0tmd3k",
        "Jazz": "https://www.youtube.com/watch?v=3tmd-ClpJxA",
        "Classical": "https://www.youtube.com/watch?v=GRxofEmo3HA",
        "Pop": "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"
    }
    if music_pred in playlist_links:
        st.markdown(f"[🎵 Listen Here]({playlist_links[music_pred]})")

# --- Optional: Visualizations ---
st.subheader("📊 Mood vs Task Heatmap")
df = pd.read_csv("dataset/smart_study_data.csv")
heatmap_data = pd.crosstab(df['Mood'], df['Task'])
fig, ax = plt.subplots()
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
