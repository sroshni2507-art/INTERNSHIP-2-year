import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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
    music_map = {
        ("Sad", "Relaxing"): "Calm Acoustic / Ambient",
        ("Energetic", "Workout"): "High BPM EDM / Hip-Hop",
        ("Calm", "Studying"): "Lo-fi / Instrumental",
        ("Stressed", "Coding"): "Low-lyric Electronic"
    }
    music_pred = music_map.get((mood, activity), "Soft Background Music")

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

    # ---- TIME-BASED SUGGESTIONS ----
    time_suggestion = ""
    if 5 <= time_of_day <= 11:
        time_suggestion = "Morning Tip: Start your day with focused work or light exercise."
    elif 12 <= time_of_day <= 17:
        time_suggestion = "Afternoon Tip: Take a productive break or do creative tasks."
    else:
        time_suggestion = "Evening Tip: Relax or do light meditation before bed."

    # ---- OUTPUT ----
    st.subheader("✅ Recommendations")
    st.success(f"📋 Recommended Task: **{task_pred}**")
    st.info(f"🎧 Recommended Music Type: **{music_pred}**")
    st.info(f"⏰ {time_suggestion}")

    # ---- OPTIONAL MUSIC LINKS ----
    playlist_links = {
        "Lo-fi / Instrumental": "https://www.youtube.com/watch?v=3AtDnEC4zak",
        "Low-lyric Electronic": "https://www.youtube.com/watch?v=HMnrl0tmd3k",
        "Calm Acoustic / Ambient": "https://www.youtube.com/watch?v=GRxofEmo3HA",
        "High BPM EDM / Hip-Hop": "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"
    }

    if music_pred in playlist_links:
        st.markdown(f"[🎵 Listen Here]({playlist_links[music_pred]})")

    # ---- TASK COMPLETION TRACKER ----
    st.subheader("✔️ Task Completion")
    task_status = st.radio(f"Did you complete **{task_pred}**?", ("Not yet", "Done", "Skipped"))

    # ---------------- HISTORY LOG ----------------
    HISTORY_PATH = os.path.join(BASE_DIR, "dataset", "recommendation_history.csv")
    if not os.path.exists(os.path.join(BASE_DIR, "dataset")):
        os.makedirs(os.path.join(BASE_DIR, "dataset"))

    history_entry = pd.DataFrame([{
        "Mood": mood,
        "Activity": activity,
        "Goal": goal,
        "Time": time_of_day,
        "Recommended Task": task_pred,
        "Music": music_pred,
        "Status": task_status
    }])

    if os.path.exists(HISTORY_PATH):
        history_entry.to_csv(HISTORY_PATH, mode='a', header=False, index=False)
    else:
        history_entry.to_csv(HISTORY_PATH, index=False)

# ---------------- VISUALIZATION ----------------
st.subheader("📊 Mood vs Task Heatmap")

# Path to dataset CSV
DATA_PATH = os.path.join(BASE_DIR, "dataset", "smart_study_data.csv")

# Create sample CSV if missing
if not os.path.exists(DATA_PATH):
    sample_data = pd.DataFrame({
        "Mood": ["Happy", "Sad", "Calm", "Energetic", "Stressed", "Calm", "Happy",
                 "Energetic", "Sad", "Happy", "Calm", "Stressed", "Energetic", "Happy",
                 "Calm", "Sad", "Energetic", "Happy", "Calm", "Stressed"],
        "Task": ["Study", "Relax", "Reading", "Workout", "Coding", "Meditation", "Exercise",
                 "Run", "Watch Movie", "Write", "Yoga", "Debug", "Cycling", "Presentation",
                 "Research", "Nap", "Gym", "Project", "Painting", "Organize"]
    })
    sample_data.to_csv(DATA_PATH, index=False)

try:
    df = pd.read_csv(DATA_PATH)

    if "Mood" in df.columns and "Task" in df.columns:
        # Interactive Plotly heatmap
        heatmap_data = pd.crosstab(df["Mood"], df["Task"])
        fig = px.imshow(
            heatmap_data,
            text_auto=True,
            color_continuous_scale='RdBu',
            labels=dict(x="Task", y="Mood", color="Count"),
            title="Mood vs Task Heatmap (Interactive)"
        )
        st.plotly_chart(fig)

    else:
        st.warning("Columns 'Mood' or 'Task' not found in dataset.")
except Exception as e:
    st.error(f"Error loading or processing CSV: {e}")

# ---------------- HISTORY TABLE & DOWNLOAD ----------------
st.subheader("📝 Recommendation History")

if os.path.exists(HISTORY_PATH):
    history_df = pd.read_csv(HISTORY_PATH)
    st.dataframe(history_df)

    # Download button
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Recommendation History",
        data=csv,
        file_name='recommendation_history.csv',
        mime='text/csv'
    )

# ---------------- ANALYTICS ----------------
st.subheader("📊 Mood & Task Analytics")

if os.path.exists(HISTORY_PATH):
    # Most common moods
    mood_counts = history_df['Mood'].value_counts()
    st.bar_chart(mood_counts)

    # Most recommended tasks
    task_counts = history_df['Recommended Task'].value_counts()
    st.bar_chart(task_counts)
