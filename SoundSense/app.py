import streamlit as st
import librosa
import numpy as np
import pickle
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import io
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | ML SonicSense",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MODEL LOADING LOGIC (From PDF) ---
@st.cache_resource
def load_ml_models():
    try:
        # PDF-ல் நீங்கள் உருவாக்கிய மாடல்கள் மற்றும் என்கோடர்களை இங்கே லோட் செய்கிறோம்
        with open('nb_task.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        with open('knn_music.pkl', 'rb') as f:
            knn_model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return nb_model, knn_model, encoders, True
    except:
        return None, None, None, False

nb_model, knn_model, encoders, model_loaded = load_ml_models()

# --- 3. PREMIUM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=Poppins:wght@400;700&display=swap');

    .stApp {
        background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .main-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.82); z-index: -1;
    }
    [data-testid="stSidebar"] {
        background-color: #050510 !important;
        border-right: 3px solid #ff00c1 !important;
    }
    [data-testid="stSidebar"] * {
        color: #ff00c1 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 800 !important;
    }
    .hero-header {
        text-align: center; padding: 40px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 35px; border: 2px solid #ff00c1;
        backdrop-filter: blur(15px); margin-bottom: 30px;
    }
    .company-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 5rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 12px;
    }
    .glass-card {
        background: rgba(10, 10, 20, 0.95);
        padding: 30px; border-radius: 25px;
        border: 1px solid rgba(255, 0, 193, 0.4);
        margin-bottom: 25px;
    }
    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; }
    p, label { font-size: 1.3rem !important; color: white !important; font-family: 'Poppins', sans-serif; }
    .stButton>button {
        background: linear-gradient(45deg, #ff00c1, #00d2ff);
        color: white !important;
        border-radius: 50px;
        padding: 15px 45px;
        font-weight: 900;
        width: 100%;
    }
    </style>
    <div class="main-overlay"></div>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    menu = st.radio("SELECT MODULE:", ["🏠 Dashboard", "🧠 Mood Spotify AI (ML)", "🎙️ Creative Studio", "♿ Hearing Assist"])
    st.write("---")
    if model_loaded:
        st.success("🤖 ML MODELS: LOADED")
    else:
        st.warning("⚠️ ML MODELS: NOT FOUND")

# --- 5. TOP HEADER ---
st.markdown("""
    <div class="hero-header">
        <h1 class="company-title">TECHNOVA SOLUTION</h1>
        <p style="letter-spacing: 6px; color:#92fe9d; font-size:1.6rem; font-weight:700;">SONICSENSE ULTRA PRO</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MODULES ---

# --- MOOD SPOTIFY AI (PDF ML INTEGRATED) ---
if "Mood Spotify AI" in menu:
    st.markdown("<div class='glass-card'><h3>🧠 AI Mood & Task Intelligence</h3></div>", unsafe_allow_html=True)
    
    # PDF-ல் உள்ள கேட்டகிரிகள்
    mood_list = ["Calm", "Stressed", "Energetic", "Sad"]
    activity_list = ["Studying", "Coding", "Workout", "Relaxing", "Sleeping"]
    goal_list = ["Focus", "Relaxation", "Energy Boost"]
    
    # Spotify Playlist Mapping
    spotify_playlists = {
        "Lo-Fi": "37i9dQZF1DX8UebicO9uaR",
        "Electronic": "37i9dQZF1DX6J5NfMJS675",
        "Jazz": "37i9dQZF1DXbITWG1ZUBIB",
        "Classical": "37i9dQZF1DX8u97vXmZp9v",
        "Pop": "37i9dQZF1DXcBWIGvYBM3s",
        "Rock": "37i9dQZF1DX8FwnS9Y9v9v",
        "Ambient": "37i9dQZF1DX3YSRmBhyV9O",
        "Hip-Hop": "37i9dQZF1DX0XUsKG7P9v8"
    }

    col1, col2 = st.columns([1, 1.4])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("Current Mood:", mood_list)
        u_activity = st.selectbox("What are you doing?", activity_list)
        u_goal = st.selectbox("What is your goal?", goal_list)
        u_time = datetime.now().hour # ஆட்டோமேட்டிக்காக தற்போதைய நேரத்தை எடுக்கும்
        
        if st.button("🚀 PREDICT & LAUNCH"):
            if model_loaded:
                # 1. Encoding inputs (PDF லாஜிக்)
                m_enc = encoders['le_mood'].transform([u_mood])[0]
                a_enc = encoders['le_activity'].transform([u_activity])[0]
                g_enc = encoders['le_goal'].transform([u_goal])[0]
                
                X_user = np.array([[m_enc, a_enc, u_time, g_enc]])
                
                # 2. ML Prediction
                task_p = encoders['le_task'].inverse_transform(nb_model.predict(X_user))[0]
                music_p = encoders['le_music'].inverse_transform(knn_model.predict(X_user))[0]
                
                st.session_state.ml_task = task_p
                st.session_state.ml_music = music_p
                st.balloons()
            else:
                # Fallback if models not found
                st.session_state.ml_task = "Complete Project"
                st.session_state.ml_music = "Lo-Fi"
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if 'ml_task' in st.session_state:
            st.markdown(f"""
                <div class='glass-card' style='border: 2px solid #00d2ff;'>
                    <h3 style='text-align:center;'>AI Recommendation</h3>
                    <p style='text-align:center; color:#92fe9d;'>Recommended Task: <b>{st.session_state.ml_task}</b></p>
                    <p style='text-align:center;'>Music Vibe: <b>{st.session_state.ml_music}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Spotify Integration based on ML Prediction
            genre = st.session_state.ml_music
            pid = spotify_playlists.get(genre, "37i9dQZF1DX8UebicO9uaR")
            embed_url = f"https://open.spotify.com/embed/playlist/{pid}?utm_source=generator&theme=0"
            components.iframe(embed_url, height=380, scrolling=False)
        else:
            st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?q=80&w=800", use_container_width=True)

# --- மற்ற மாட்யூல்கள் (முன்பு போலவே இருக்கும்) ---
elif "Dashboard" in menu:
    st.snow()
    st.markdown("<div class='glass-card'><h2>Technova Dashboard</h2><p>PDF-ல் உள்ள ML மாடல்கள் வெற்றிகரமாக SonicSense Pro-உடன் இணைக்கப்பட்டுள்ளது.</p></div>", unsafe_allow_html=True)

elif "Creative Studio" in menu:
    st.markdown("<div class='glass-card'><h3>🎙️ Creative AI Studio</h3></div>", unsafe_allow_html=True)
    voice = st.audio_input("Record voice to generate music:")
    if voice and st.button("✨ TRANSFORM"):
        st.balloons()

elif "Hearing Assist" in menu:
    st.markdown("<div class='glass-card'><h3>♿ Inclusive Hearing Assist</h3></div>", unsafe_allow_html=True)
    up = st.file_uploader("Upload audio", type=["mp3", "wav"])
