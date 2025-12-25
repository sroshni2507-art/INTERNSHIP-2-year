import streamlit as st
import librosa
import librosa.display
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile
import time
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TECHNOVA SOLUTION | Pro AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE FOR BPM TAPPER & ML ---
if 'pred_task' not in st.session_state:
    st.session_state.pred_task = None
if 'pred_genre' not in st.session_state:
    st.session_state.pred_genre = None
if 'taps' not in st.session_state:
    st.session_state.taps = []
if 'bpm_val' not in st.session_state:
    st.session_state.bpm_val = 0

# --- 3. SMART PATH LOGIC FOR ML FILES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    nb_path = os.path.join(BASE_DIR, 'nb_task.pkl')
    knn_path = os.path.join(BASE_DIR, 'knn_music.pkl')
    enc_path = os.path.join(BASE_DIR, 'encoders.pkl')
    if os.path.exists(nb_path) and os.path.exists(knn_path) and os.path.exists(enc_path):
        try:
            with open(nb_path, 'rb') as f: nb_model = pickle.load(f)
            with open(knn_path, 'rb') as f: knn_model = pickle.load(f)
            with open(enc_path, 'rb') as f: encoders = pickle.load(f)
            return nb_model, knn_model, encoders, True
        except: return None, None, None, False
    else: return None, None, None, False

nb_model, knn_model, encoders, is_ml_ready = load_models()

# --- 4. ADVANCED CSS (PINK & NEON THEME) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=Poppins:wght@400;700;900&display=swap');
    
    .stApp { background: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop"); background-size: cover; background-attachment: fixed; }
    .main-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.82); z-index: -1; }
    
    [data-testid="stSidebar"] { background-color: #050510 !important; border-right: 3px solid #ff00c1 !important; }
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #ff00c1 !important; font-family: 'Poppins', sans-serif !important; font-weight: 900 !important; font-size: 1.15rem !important;
    }

    .hero-header { text-align: center; padding: 40px; background: rgba(255, 255, 255, 0.05); border-radius: 35px; border: 2px solid #ff00c1; backdrop-filter: blur(15px); margin-bottom: 30px; }
    .company-title { font-family: 'Orbitron', sans-serif; font-size: 5rem !important; font-weight: 900; background: linear-gradient(90deg, #ff00c1, #00d2ff, #92fe9d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 12px; }
    .glass-card { background: rgba(10, 10, 20, 0.95); padding: 30px; border-radius: 25px; border: 1px solid rgba(255, 0, 193, 0.4); margin-bottom: 25px; }
    
    h2, h3 { color: #00d2ff !important; font-family: 'Orbitron', sans-serif; font-size: 2.5rem !important; }
    p, label { font-size: 1.35rem !important; color: white !important; font-family: 'Poppins', sans-serif; font-weight: 600; }

    .stButton>button { background: linear-gradient(45deg, #ff00c1, #00d2ff); color: white !important; border-radius: 50px; padding: 15px 45px; font-weight: 900; width: 100%; border: none; box-shadow: 0 0 30px rgba(255, 0, 193, 0.4); }
    </style>
    <div class="main-overlay"></div>
    """, unsafe_allow_html=True)

# --- 5. AUDIO LOGIC FUNCTIONS ---

def voice_to_music(audio, sr):
    hop_length = 512
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), hop_length=hop_length)
    f0 = np.nan_to_num(f0)
    f0_stretched = np.repeat(f0, hop_length)
    if len(f0_stretched) < len(audio): f0_stretched = np.pad(f0_stretched, (0, len(audio) - len(f0_stretched)))
    else: f0_stretched = f0_stretched[:len(audio)]
    phase = np.cumsum(2 * np.pi * f0_stretched / sr)
    music = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)
    max_val = np.max(np.abs(music))
    return music / max_val if max_val > 0 else music

def text_to_song_logic(text):
    sr = 44100
    words = text.split()
    note_dur = 0.5
    full_song = np.array([])
    scale = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25] # Pentatonic
    for word in words:
        freq = scale[len(word) % len(scale)]
        t = np.linspace(0, note_dur, int(sr * note_dur))
        envelope = np.exp(-3 * t / note_dur) 
        note = 0.5 * np.sin(2 * np.pi * freq * t) * envelope
        full_song = np.concatenate([full_song, note])
    return full_song, sr

def voice_morpher(y, sr, effect):
    if effect == "Child 👶": return librosa.effects.pitch_shift(y, sr=sr, n_steps=5)
    if effect == "Villain 👿": return librosa.effects.pitch_shift(y, sr=sr, n_steps=-5)
    if effect == "Robot 🤖":
        y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        return np.clip(y_shift * 1.5, -1, 1)
    return y

def lyric_assistant(theme):
    lyrics = {
        "Love": ["My heart skips a beat for you", "Walking under the stars together", "In your eyes, I found my world", "Love is a melody we sing"],
        "Space": ["Floating in the infinite dark", "Galaxy of dreams in my head", "Riding the comet through the void", "Alien signals calling me home"],
        "Rain": ["Droplets dancing on my window", "The smell of earth when sky cries", "Walking lonely in the storm", "Wash away the pain of yesterday"]
    }
    return lyrics.get(theme, ["Sing your heart out...", "The music flows through you..."])

# --- 6. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#00d2ff !important;'>TECHNOVA</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3659/3659784.png", width=120)
    st.write("---")
    if is_ml_ready: st.success("✅ AI ENGINE: ACTIVE")
    else: st.error("⚠️ ML FILES MISSING")
    
    choice = st.radio("SELECT MODULE:", ["🏠 Dashboard", "❄️❄️❄️ Mood AI", "🎭 Voice Morpher", "🎨🎨🎨 Creative Studio", "♿ Hearing Assist", "🎹 BPM Tapper"])

# --- 7. HEADER ---
st.markdown("""<div class="hero-header"><h1 class="company-title">TECHNOVA SOLUTION</h1><p style="letter-spacing: 6px; color:#92fe9d; font-size:1.6rem; font-weight:700;">SONICSENSE ULTRA PRO</p></div>""", unsafe_allow_html=True)

# --- 8. MODULES ---

# --- DASHBOARD ---
if choice == "🏠 Dashboard":
    st.snow()
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("<div class='glass-card'><h2>The Future of Audio</h2><p>Technova Solution bridges the gap between sound and technology. Explore our smart prediction and creative tools designed for everyone.</p></div>", unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=500&h=500&fit=crop", use_container_width=True)

# --- MOOD AI ---
elif choice == "❄️❄️❄️ Mood AI":
    st.markdown("<div class='glass-card'><h3>🧠 AI Mood & Task Prediction</h3></div>", unsafe_allow_html=True)
    genre_search_map = {"Lo-Fi": "lofi focus music", "Electronic": "electronic workout music", "Jazz": "smooth jazz music", "Classical": "classical focus music", "Pop": "top pop hits", "Ambient": "ambient calm music", "Rock": "rock energy music"}
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        u_mood = st.selectbox("Current Mood:", ["Calm", "Stressed", "Energetic", "Sad"])
        u_act = st.selectbox("Activity:", ["Studying", "Coding", "Workout", "Relaxing", "Sleeping"])
        if st.button("🚀 PREDICT & SUGGEST"):
            if is_ml_ready:
                m_enc = encoders['le_mood'].transform([u_mood])[0]
                a_enc = encoders['le_activity'].transform([u_act])[0]
                X = np.array([[m_enc, a_enc, datetime.now().hour, 0]])
                st.session_state.pred_task = encoders['le_task'].inverse_transform(nb_model.predict(X))[0]
                st.session_state.pred_genre = encoders['le_music'].inverse_transform(knn_model.predict(X))[0]
            else:
                st.session_state.pred_task = "Focus Session"; st.session_state.pred_genre = "Lo-Fi"
            st.balloons(); st.snow()
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        if st.session_state.pred_genre:
            genre = st.session_state.pred_genre
            search_url = f"https://open.spotify.com/search/{genre_search_map.get(genre, 'lofi').replace(' ', '%20')}"
            st.markdown(f"<div class='glass-card' style='text-align:center; border: 2px solid #1DB954;'><h3>🎧 Recommendation</h3><p><b>Task:</b> {st.session_state.pred_task}</p><p><b>Music:</b> {genre}</p><br><a href='{search_url}' target='_blank'><button style='background:linear-gradient(45deg,#1DB954,#1ed760); color:white; padding:15px 30px; border:none; border-radius:50px; font-weight:800; cursor:pointer;'>🔗 OPEN IN SPOTIFY</button></a></div>", unsafe_allow_html=True)

# --- VOICE MORPHER (FEATURE 1) ---
elif choice == "🎭 Voice Morpher":
    st.markdown("<div class='glass-card'><h3>🎭 AI Voice Changer</h3></div>", unsafe_allow_html=True)
    v_up = st.file_uploader("Upload Voice File:", type=["wav", "mp3"], key="morph_up")
    effect = st.selectbox("Choose Character:", ["Child 👶", "Villain 👿", "Robot 🤖"])
    if v_up and st.button("✨ TRANSFORM VOICE"):
        y, sr = librosa.load(v_up)
        morphed = voice_morpher(y, sr, effect)
        st.audio(morphed, sample_rate=sr)
        st.success(f"Character {effect} Applied Successfully!")

# --- CREATIVE STUDIO ---
elif choice == "🎨🎨🎨 Creative Studio":
    st.markdown("<div class='glass-card'><h3>🎙️ Creative AI Studio</h3></div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎤 RECORD LIVE", "📤 UPLOAD FILE", "✍️ TEXT TO SONG", "🎵 AI BGM MIXER", "🎤 KARAOKE PRO", "🎼 VISUALIZER"])
    
    with tab1:
        v = st.audio_input("Record voice to convert:")
        if v and st.button("✨ TRANSFORM RECORDING"):
            y, sr = librosa.load(v); processed = voice_to_music(y, sr)
            st.audio(processed, sample_rate=sr); st.balloons()
    
    with tab2:
        up = st.file_uploader("Upload Audio (MP3/WAV):", type=["mp3","wav"], key="creative_up")
        if up and st.button("🚀 TRANSFORM UPLOAD"):
            y, sr = librosa.load(up); processed = voice_to_music(y, sr)
            st.audio(processed, sample_rate=sr); st.balloons()
    
    with tab3: # LYRICS AI (FEATURE 4)
        st.write("Need inspiration? Generate lyrics based on a theme.")
        theme = st.selectbox("Choose Theme:", ["Love", "Space", "Rain"])
        if st.button("✍️ GENERATE LYRICS"):
            lines = lyric_assistant(theme)
            for line in lines: st.write(f"🎶 *{line}*")
        st.write("---")
        lyrics_txt = st.text_area("Or Input Your Own Lyrics:")
        if lyrics_txt and st.button("🎵 GENERATE THEME FROM TEXT"):
            song, sr_s = text_to_song_logic(lyrics_txt)
            st.audio(song, sample_rate=sr_s); st.balloons()

    with tab4:
        st.write("Mix your voice with background music.")
        v_file_mix = st.file_uploader("Upload Voice File:", type=["wav", "mp3"], key="v_mix")
        b_file_mix = st.file_uploader("Upload BGM File:", type=["wav", "mp3"], key="b_mix")
        if v_file_mix and b_file_mix and st.button("🎚️ MIX VOICE & BGM"):
            with st.spinner("Processing Fusion..."):
                v_data, v_sr = librosa.load(v_file_mix, sr=None)
                b_data, b_sr = librosa.load(b_file_mix, sr=v_sr)
                b_data = b_data * 0.17 
                min_len = min(len(v_data), len(b_data))
                mixed = v_data[:min_len] + b_data[:min_len]
                final_path = "final_song.wav"
                sf.write(final_path, mixed, v_sr)
                st.audio(final_path)
                st.success("final_song.wav ready ✅")
                st.balloons()

    with tab5: # KARAOKE PRO (FEATURE 2)
        st.write("Extract background music (Lite version).")
        k_up = st.file_uploader("Upload Song File:", type=["wav", "mp3"], key="karaoke_up")
        if k_up and st.button("🎸 EXTRACT INSTRUMENTAL"):
            y, sr = librosa.load(k_up)
            # Center channel subtraction (vocal reduction logic)
            y_harmonic = librosa.effects.harmonic(y)
            st.audio(y_harmonic, sample_rate=sr)
            st.info("Vocal reduction applied. Ready for Karaoke!")

    with tab6: # VISUALIZER (FEATURE 3)
        st.write("Professional Audio Waveform Analysis")
        vis_up = st.file_uploader("Upload Audio for Visualization:", type=["wav", "mp3"], key="vis_up")
        if vis_up:
            y, sr = librosa.load(vis_up)
            fig, ax = plt.subplots(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax, color="#ff00c1")
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            st.pyplot(fig)

# --- HEARING ASSIST ---
elif choice == "♿ Hearing Assist":
    st.markdown("<div class='glass-card'><h3>♿ Inclusive Hearing Assist</h3><p>Optimizing sound frequencies for vibrations.</p></div>", unsafe_allow_html=True)
    up_h = st.file_uploader("Upload audio for frequency shift", type=["mp3", "wav"], key="hearing_up")
    if up_h:
        y, sr = librosa.load(up_h)
        shift = st.slider("Frequency Sensitivity (Lower pitch = more vibration)", -12, 0, -8)
        if st.button("🔊 OPTIMIZE Pattern"):
            st.snow()
            y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift)
            st.audio(y_shift * 1.5, sample_rate=sr)
            st.success("Sound optimized successfully.")

# --- BPM TAPPER (FEATURE 5) ---
elif choice == "🎹 BPM Tapper":
    st.markdown("<div class='glass-card'><h3>🎹 Virtual BPM Tapper</h3><p>Tap the button in rhythm with a song to find its Tempo.</p></div>", unsafe_allow_html=True)
    if st.button("🥁 TAP TO THE BEAT"):
        st.session_state.taps.append(time.time())
        if len(st.session_state.taps) > 1:
            diffs = np.diff(st.session_state.taps[-8:]) # Consider last 8 taps
            avg_diff = np.mean(diffs)
            st.session_state.bpm_val = 60 / avg_diff
    
    st.metric("Estimated Tempo (BPM)", f"{int(st.session_state.bpm_val)}")
    
    if st.button("🔄 Reset Tapper"):
        st.session_state.taps = []
        st.session_state.bpm_val = 0
        st.rerun()
