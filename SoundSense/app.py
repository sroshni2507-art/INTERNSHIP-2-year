import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="TECHNOVA SOLUTION | SonicSense",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Poppins:wght@400;700&display=swap');

.stApp {
    background: radial-gradient(circle at top, #0f0f1a, #000);
    color: white;
}

[data-testid="stSidebar"] {
    background: #050510;
    border-right: 2px solid #ff00c1;
}

[data-testid="stSidebar"] * {
    color: #ff00c1 !important;
    font-weight: 700;
}

.glass {
    background: rgba(255,255,255,0.06);
    padding: 30px;
    border-radius: 25px;
    border: 1px solid #ff00c155;
    margin-bottom: 20px;
}

.stButton>button {
    background: linear-gradient(45deg,#ff00c1,#00d2ff);
    border-radius: 40px;
    font-size: 18px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ================= AUDIO LOGIC =================
def voice_to_music(audio, sr):
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    music = np.sin(phase)
    return music / (np.max(np.abs(music)) + 1e-6)

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🎧 TECHNOVA")
    menu = st.radio(
        "SELECT MENU",
        ["🏠 Dashboard", "🧠 Mood Spotify AI", "🎙️ Creative Studio", "♿ Hearing Assist"]
    )
    st.success("AI ENGINE : ONLINE")

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;'>TECHNOVA SOLUTION</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>SONICSENSE ULTRA PRO</p>", unsafe_allow_html=True)

# ================= DASHBOARD =================
if menu == "🏠 Dashboard":
    st.markdown("<div class='glass'><h2>AI Audio Intelligence</h2><p>Voice → Music • Mood → Focus • Sound → Vision</p></div>", unsafe_allow_html=True)

# ================= MOOD SPOTIFY AI (FIXED) =================
elif menu == "🧠 Mood Spotify AI":
    st.markdown("<div class='glass'><h2>🎶 Mood Based Music</h2></div>", unsafe_allow_html=True)

    mood_query = {
        "Energetic 🔥": "energetic workout music",
        "Calm 🌊": "calm lofi music",
        "Focused 🎯": "deep focus music",
        "Stressed 🧘": "stress relief music",
        "Devotional ✨": "bhakti devotional music"
    }

    mood = st.selectbox("HOW ARE YOU FEELING?", list(mood_query.keys()))
    goal = st.text_input("YOUR GOAL TODAY", "Finish Internship Project")

    if st.button("🚀 LAUNCH SESSION"):
        st.balloons()

        search_term = mood_query[mood]
        embed_url = f"https://open.spotify.com/embed/search/{search_term.replace(' ','%20')}"

        st.markdown(f"### 🎧 Vibe : {mood}")
        components.iframe(embed_url, height=380)

        st.markdown(
            f"""
            <a href="https://open.spotify.com/search/{search_term.replace(' ','%20')}" target="_blank">
            <button style="width:100%;padding:15px;border-radius:30px;
            background:#1DB954;color:white;font-size:18px;">
            🔗 OPEN IN SPOTIFY APP
            </button></a>
            """,
            unsafe_allow_html=True
        )

# ================= CREATIVE STUDIO =================
elif menu == "🎙️ Creative Studio":
    st.markdown("<div class='glass'><h2>🎤 Voice to Music</h2></div>", unsafe_allow_html=True)

    audio = st.audio_input("Record your voice")
    if audio:
        y, sr = librosa.load(audio)
        if st.button("🎵 TRANSFORM"):
            music = voice_to_music(y, sr)
            st.audio(music, sample_rate=sr)

# ================= HEARING ASSIST =================
elif menu == "♿ Hearing Assist":
    st.markdown("<div class='glass'><h2>♿ Hearing Assist</h2><p>Visual vibration patterns</p></div>", unsafe_allow_html=True)

    file = st.file_uploader("Upload audio", type=["wav","mp3"])
    if file:
        y, sr = librosa.load(file)
        rms = librosa.feature.rms(y=y)[0]
        fig, ax = plt.subplots()
        ax.plot(rms)
        st.pyplot(fig)
