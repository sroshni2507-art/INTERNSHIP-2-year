import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="SonicSense Pro", layout="wide")

# --- CUSTOM ENHANCED CSS ---
st.markdown("""
    <style>
    .stApp { background: #0f0c29; background: linear-gradient(to right, #0f0c29, #302b63, #24243e); color: white; }
    .stButton>button { 
        background: linear-gradient(45deg, #FF512F, #DD2476); 
        color: white; border-radius: 25px; border: none; font-size: 18px; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.05); box-shadow: 0 10px 20px rgba(0,0,0,0.3); }
    .reportview-container .main .block-container { padding: 3rem; }
    h1, h2, h3 { color: #00d2ff; }
    .card { background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 SonicSense Pro: The Future of Inclusive Sound")

# Sidebar
st.sidebar.title("💎 SonicSense Menu")
menu = ["🏠 Dashboard", "🎤 Voice → Instrument", "🧠 Mood AI & Spotify", "🌈 Sensory Room (Deaf Support)", "🎬 Movie Accessibility"]
choice = st.sidebar.radio("Navigate", menu)

# ---------------------------------------------------------
# MODULE: MOOD AI & SPOTIFY (Work Activity Boost)
# ---------------------------------------------------------
if choice == "🧠 Mood AI & Spotify":
    st.header("Personalized Productivity & Spotify Sync")
    
    col1, col2 = st.columns(2)
    with col1:
        mood = st.selectbox("How's your mood?", ["Stressed", "Calm", "Focused", "Energetic"])
        goal = st.text_input("What are you working on?", "Coding my AI project")
    
    if st.button("Generate My Flow State"):
        st.balloons()
        
        # Recommendations Logic
        recs = {
            "Stressed": ("Deep Breathing & Lo-Fi", "Lofi Beats", "https://open.spotify.com/search/lofi%20relax"),
            "Calm": ("Mindful Journaling & Classical", "Classical", "https://open.spotify.com/search/classical%20piano"),
            "Focused": ("Deep Work & Ambient Noise", "Ambient", "https://open.spotify.com/search/focus%20ambient"),
            "Energetic": ("HIIT Workout & EDM", "EDM", "https://open.spotify.com/search/workout%20edm")
        }
        task, genre, link = recs[mood]
        
        st.markdown(f"""
        <div class='card'>
            <h3>✅ Targeted Activity: {task}</h3>
            <p>Work on <b>{goal}</b> with high efficiency.</p>
            <a href='{link}' target='_blank'><button style='padding: 10px; border-radius: 10px; background: #1DB954; color: white; border: none; cursor: pointer;'>🎧 Open {genre} on Spotify</button></a>
        </div>
        """, unsafe_allow_html=True)

        # Download Feature
        plan_text = f"Mood: {mood}\nActivity: {task}\nGoal: {goal}\nMusic: {genre}"
        st.download_button("📥 Download My Productivity Plan", plan_text, file_name="flow_plan.txt")

        # Pomodoro Timer
        st.write("---")
        st.subheader("⏱️ Productivity Timer")
        if st.button("Start 25 min Deep Work Session"):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1) # Demo speed
                progress_bar.progress(i + 1)
            st.success("Session Complete! Take a break.")

# ---------------------------------------------------------
# MODULE: SENSORY ROOM (For Hearing Impaired)
# ---------------------------------------------------------
elif choice == "🌈 Sensory Room (Deaf Support)":
    st.header("Feel the Sound: Visual Sensory Experience")
    st.info("Hearing impaired users can 'watch' the music through color ripples and vibration maps.")
    
    music_file = st.file_uploader("Upload Audio to Visualize", type=['wav', 'mp3'])
    
    if music_file:
        y, sr = librosa.load(music_file)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        st.write(f"Detected Rhythm Tempo: {round(float(tempo), 1)} BPM")
        
        # Accessibility Visualization
        rms = librosa.feature.rms(y=y)[0]
        fig, ax = plt.subplots(figsize=(12, 4))
        # Changing color based on volume for sensory effect
        ax.fill_between(range(len(rms)), rms, color=plt.cm.magma(np.max(rms)*10))
        ax.set_axis_off()
        st.pyplot(fig)
        
        st.markdown("### 🌈 Sensory Flash")
        st.write("The screen background changes with the beat (Simulated below)")
        # This simulates a sensory light for deaf users
        if st.button("Start Sensory Light Sync"):
            for _ in range(5):
                st.markdown("<div style='height:50px; background:red; border-radius:10px;'></div>", unsafe_allow_html=True)
                time.sleep(0.3)
                st.markdown("<div style='height:50px; background:blue; border-radius:10px;'></div>", unsafe_allow_html=True)
                time.sleep(0.3)

# ---------------------------------------------------------
# MODULE: VOICE → INSTRUMENT (Refined with Download)
# ---------------------------------------------------------
elif choice == "🎤 Voice → Instrument":
    st.header("Creative Voice Transformer")
    v_file = st.file_uploader("Upload your hum/voice", type=['wav', 'mp3'])
    
    if v_file:
        y, sr = librosa.load(v_file)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = np.nan_to_num(f0)
        
        hop_length = 512
        f0_stretched = np.repeat(f0, hop_length)[:len(y)]
        phase = np.cumsum(2 * np.pi * f0_stretched / sr)
        music = 0.5 * np.sin(phase) + 0.2 * np.sin(2*phase)
        music[f0_stretched == 0] = 0
        
        # Output & Download
        out_buf = io.BytesIO()
        sf.write(out_buf, music, sr, format='WAV')
        
        st.audio(out_buf)
        st.download_button("💾 Download AI Created Music", out_buf, file_name="ai_music.wav")

# ---------------------------------------------------------
# MODULE: DASHBOARD (Home)
# ---------------------------------------------------------
elif choice == "🏠 Dashboard":
    st.subheader("Your AI Hub Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accessibility Level", "High", "Accessibility+")
    col2.metric("Productivity Boost", "45%", "Focus Mode")
    col3.metric("AI Music Created", "12 Tracks", "Creative")
    
    st.markdown("""
    ### Why SonicSense Pro?
    - **Productivity:** Syncs with Spotify and uses Pomodoro for work activity.
    - **Inclusion:** Sensory Room helps the hearing impaired feel the rhythm through light and visuals.
    - **Creativity:** Turn any sound into a professional instrument track.
    """)
