import streamlit as st
import pickle
import numpy as np
import pandas as pd
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
st.set_page_config(page_title="🎵 Adaptive Music Companion", layout="wide")

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🎵 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Recommendations", "Mood vs Task Heatmap", "History", "Analytics"])

# ---------------- DATASET PATHS ----------------
DATA_PATH = os.path.join(BASE_DIR, "dataset", "smart_study_data.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "dataset", "recommendation_history.csv")
if not os.path.exists(os.path.join(BASE_DIR, "dataset")):
    os.makedirs(os.path.join(BASE_DIR, "dataset"))

# ---------------- CREATE SAMPLE DATA IF MISSING ----------------
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

# ---------------- PAGE: HOME ----------------
if page == "Home":
    st.markdown("<h1 style='color:#4CAF50;'>🎵 Adaptive Music & Productivity Companion</h1>", unsafe_allow_html=True)
    st.markdown("""
        Welcome! This app recommends **tasks** and **music** based on your **mood, activity, time, and goal**.  
        Use the sidebar to navigate between sections.
    """)

# ---------------- PAGE: RECOMMENDATIONS ----------------
elif page == "Recommendations":
    st.markdown("<h2 style='color:#FF5733;'>📋 Get Your Personalized Recommendations</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        mood = st.selectbox("Select Your Mood:", encoders["le_mood"].classes_)
        activity = st.selectbox("Select Your Activity:", encoders["le_activity"].classes_)
        time_of_day = st.slider("Select Time of Day (Hour 0–23):", 0, 23, 14)

    with col2:
        goal = st.selectbox("Select Your Goal:", encoders["le_goal"].classes_)

    if st.button("Get Recommendation"):
        # ---- Music Recommendation ----
        music_map = {
            ("Sad", "Relaxing"): "Calm Acoustic / Ambient",
            ("Energetic", "Workout"): "High BPM EDM / Hip-Hop",
            ("Calm", "Studying"): "Lo-fi / Instrumental",
            ("Stressed", "Coding"): "Low-lyric Electronic"
        }
        music_pred = music_map.get((mood, activity), "Soft Background Music")

        # ---- Encode input ----
        X_user = np.array([[ 
            encoders["le_mood"].transform([mood])[0],
            encoders["le_activity"].transform([activity])[0],
            time_of_day,
            encoders["le_goal"].transform([goal])[0]
        ]])

        # ---- Task Prediction ----
        task_pred = encoders["le_task"].inverse_transform(nb_task.predict(X_user))[0]

        # ---- Time-Based Tip ----
        if 5 <= time_of_day <= 11:
            time_tip = "🌅 Morning Tip: Start your day with focused work or light exercise."
        elif 12 <= time_of_day <= 17:
            time_tip = "☀️ Afternoon Tip: Take a productive break or do creative tasks."
        else:
            time_tip = "🌙 Evening Tip: Relax or do light meditation before bed."

        # ---- Display Recommendations in Columns ----
        col1_card, col2_card = st.columns(2)
        with col1_card:
            st.markdown(f"<div style='background-color:#fce4ec; padding:10px; border-radius:10px;'>📋 Task Recommendation:<br><b>{task_pred}</b></div>", unsafe_allow_html=True)
        with col2_card:
            st.markdown(f"<div style='background-color:#e8f5e9; padding:10px; border-radius:10px;'>🎧 Music Recommendation:<br><b>{music_pred}</b></div>", unsafe_allow_html=True)

        st.info(time_tip)

        # ---- Music Embed ----
        playlist_links = {
            "Lo-fi / Instrumental": "https://www.youtube.com/embed/3AtDnEC4zak",
            "Low-lyric Electronic": "https://www.youtube.com/embed/HMnrl0tmd3k",
            "Calm Acoustic / Ambient": "https://www.youtube.com/embed/GRxofEmo3HA",
            "High BPM EDM / Hip-Hop": "https://www.youtube.com/embed/fJ9rUzIMcZQ"
        }
        if music_pred in playlist_links:
            st.markdown("🎵 Listen to your recommended playlist:")
            st.components.v1.html(f"""
            <iframe width="300" height="80" src="{playlist_links[music_pred]}" frameborder="0" allowfullscreen></iframe>
            """, height=100)

        # ---- Task Completion Tracker ----
        task_status = st.radio(f"Did you complete **{task_pred}**?", ("Not yet", "Done", "Skipped"))

        # ---- Save Recommendation ----
        history_entry = pd.DataFrame([{
            "Mood": mood,
            "Activity": activity,
            "Goal": goal,
            "Time": time_of_day,
            "Recommended Task": task_pred,
            "Music": music_pred,
            "Status": task_status
        }])
        history_entry.to_csv(HISTORY_PATH, mode='a', header=not os.path.exists(HISTORY_PATH), index=False)

# ---------------- PAGE: MOOD VS TASK HEATMAP ----------------
elif page == "Mood vs Task Heatmap":
    st.markdown("<h2 style='color:#673AB7;'>📊 Mood vs Task Heatmap</h2>", unsafe_allow_html=True)
    try:
        df = pd.read_csv(DATA_PATH)
        heatmap_data = pd.crosstab(df["Mood"], df["Task"])
        fig = px.imshow(
            heatmap_data,
            text_auto=True,
            color_continuous_scale='Viridis',
            labels=dict(x="Task", y="Mood", color="Count"),
            title="🌈 Mood vs Task Heatmap"
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Failed to load heatmap: {e}")

# ---------------- PAGE: HISTORY ----------------
elif page == "History":
    st.markdown("<h2 style='color:#FF9800;'>📝 Recommendation History</h2>", unsafe_allow_html=True)
    try:
        history_df = pd.read_csv(HISTORY_PATH, on_bad_lines='skip')
        if not history_df.empty:
            # Add emojis for tasks
            task_emojis = {
                "Study": "📚",
                "Workout": "🏋️‍♂️",
                "Meditation": "🧘‍♀️",
                "Coding": "💻",
                "Relax": "😌",
                "Reading": "📖",
                "Exercise": "🏃‍♂️",
                "Yoga": "🧘",
                "Project": "📁",
                "Painting": "🎨",
                "Nap": "😴",
                "Debug": "🐞"
            }
            history_df["Task Emoji"] = history_df["Recommended Task"].map(task_emojis)
            st.dataframe(history_df[["Mood", "Recommended Task", "Task Emoji", "Status"]])

            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download History",
                data=csv,
                file_name='recommendation_history.csv',
                mime='text/csv'
            )
        else:
            st.info("No history available yet.")
    except FileNotFoundError:
        st.info("No history available yet.")
    except Exception as e:
        st.error(f"Error loading history: {e}")

# ---------------- PAGE: ANALYTICS ----------------
elif page == "Analytics":
    st.markdown("<h2 style='color:#009688;'>📊 Mood & Task Analytics</h2>", unsafe_allow_html=True)
    try:
        history_df = pd.read_csv(HISTORY_PATH, on_bad_lines='skip')
        if not history_df.empty:
            st.subheader("Most Common Moods")
            st.bar_chart(history_df['Mood'].value_counts())

            st.subheader("Most Recommended Tasks")
            st.bar_chart(history_df['Recommended Task'].value_counts())
        else:
            st.info("No data available yet for analytics.")
    except FileNotFoundError:
        st.info("No data available yet for analytics.")
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
