import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(__file__)

# ---------------- LOAD MODELS ----------------
try:
    nb_task = pickle.load(open(os.path.join(BASE_DIR, "nb_task.pkl"), "rb"))
    encoders = pickle.load(open(os.path.join(BASE_DIR, "encoders.pkl"), "rb"))
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(
    page_title="Adaptive Music & Productivity Companion",
    layout="wide"
)

st.title("🎵 Adaptive Music & Productivity Companion")
st.markdown(
    "Get personalized **Task** and **Music recommendations** based on your Mood, Activity, Time, and Goal!"
)

# ---------------- USER INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    mood = st.selectbox(
        "Select Your Mood:",
        encoders["le_mood"].classes_
    )

    activity = st.selectbox(
        "Select Your Activity:",
        encoders["le_activity"].classes_
    )

    time_of_day = st.slider(
        "Select Time of Day (Hour 0–23):",
        0, 23, 14
    )

with col2:
    goal = st.selectbox(
        "Select Your Goal:",
        encoders["le_goal"].classes_
    )

# ---------------- PREDICTION ----------------
if st.button("Get Recommendation"):

    # ---- RULE BASED MUSIC ----
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

    # ---- ENCODE INPUT ----
    X_user = np.array([[ 
        encoders["le_mood"].transform([mood])[0],
        encoders["le_activity"].transform([activity])[0],
        time_of_day,
        encoders["le_goal"].transform([goal])[0]
    ]])

    # ---- TASK PREDICTION ----
    task_pred = encoders["le_task"].inverse_transform(
        nb_task.predict(X_user)
    )[0]

    # ---- OUTPUT ----
    st.subheader("✅ Recommendations")
    st.success(f"📋 Recommended Task: **{task_pred}**")
    st.info(f"🎧 Recommended Music Type: **{music_pred}**")

    # ---- OPTIONAL MUSIC LINKS ----
    playlist_links = {
        "Lo-fi / Instrumental": "https://www.youtube.com/watch?v=3AtDnEC4zak",
        "Low-lyric Electronic": "https://www.youtube.com/watch?v=HMnrl0tmd3k",
        "Calm Acoustic / Ambient": "https://www.youtube.com/watch?v=GRxofEmo3HA",
        "High BPM EDM / Hip-Hop": "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"
    }

    if music_pred in playlist_links:
        st.markdown(f"[🎵 Listen Here]({playlist_links[music_pred]})")

# ---------------- VISUALIZATION ----------------
st.subheader("📊 Mood vs Task Heatmap")

# Path to dataset
DATA_PATH = os.path.join(BASE_DIR, "dataset", "smart_study_data.csv")

# Create sample CSV if missing
if not os.path.exists(DATA_PATH):
    os.makedirs(os.path.join(BASE_DIR, "dataset"), exist_ok=True)
    sample_data = pd.DataFrame({
        "Mood": ["Happy", "Sad", "Calm", "Energetic", "Stressed", "Calm", "Happy",
                 "Energetic", "Sad", "Happy", "Calm", "Stressed", "Energetic", "Happy",
                 "Calm", "Sad", "Energetic", "Happy", "Calm", "Stressed"],
        "Task": ["Study", "Relax", "Reading", "Workout", "Coding", "Meditation", "Exercise",
                 "Run", "Watch Movie", "Write", "Yoga", "Debug", "Cycling", "Presentation",
                 "Research", "Nap", "Gym", "Project", "Painting", "Organize"]
    })
    sample_data.to_csv(DATA_PATH, index=False)
    st.info("Sample dataset created as 'smart_study_data.csv' for visualization.")

# Load CSV
try:
    df = pd.read_csv(DATA_PATH)

    if "Mood" in df.columns and "Task" in df.columns:
        # Create count table per mood/task
        heatmap_data = pd.crosstab(df["Mood"], df["Task"])

        # Create labels as strings
        labels = heatmap_data.applymap(lambda x: f"{x} tasks" if x > 0 else "")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(
            heatmap_data,
            annot=labels,
            fmt="",
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Mood vs Task Heatmap", fontsize=16)
        ax.set_xlabel("Task", fontsize=12)
        ax.set_ylabel("Mood", fontsize=12)

        st.pyplot(fig)
    else:
        st.warning("Columns 'Mood' or 'Task' not found in dataset.")

except Exception as e:
    st.error(f"Error loading or processing CSV: {e}")
