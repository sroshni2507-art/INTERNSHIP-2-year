import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import speech_recognition as sr
from gtts import gTTS

# Page Config
st.set_page_config(page_title="AI Lyricist & Song Generator", layout="wide", page_icon="🎵")
st.title("🎵 AI Voice-to-Lyrics Song Maker")
st.write("Record your voice ➔ Convert to Text ➔ Edit Lyrics ➔ Generate Song with Words")

# -------- SIDEBAR SETTINGS --------
st.sidebar.header("Music Controls")
music_style = st.sidebar.selectbox("Background Style", ["EDM / Modern", "Soft Acoustic", "Retro Synth"])
vol_level = st.sidebar.slider("Voice Volume", 0.5, 2.0, 1.2)

# -------- HELPER FUNCTIONS --------

def get_text_from_voice(audio_file):
    """Converts spoken audio into written text."""
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except:
        return "I could not hear the words clearly. Please type your lyrics manually below."

def create_final_song(lyrics_text, style):
    """Generates an AI voice for the lyrics and layers it with a melody/beat."""
    # 1. Generate AI Voice (Lyrics)
    tts = gTTS(text=lyrics_text, lang='en')
    tts_buf = io.BytesIO()
    tts.write_to_fp(tts_buf)
    tts_buf.seek(0)
    
    # 2. Load the AI voice into Librosa
    y_voice, sr_rate = librosa.load(tts_buf, sr=22050)
    
    # 3. Create a Simple Background Melody/Beat
    duration = librosa.get_duration(y=y_voice, sr=sr_rate)
    t = np.arange(len(y_voice)) / sr_rate
    
    # Generate a background synth melody based on style
    if style == "EDM / Modern":
        bg_music = 0.1 * np.sign(np.sin(2 * np.pi * 440 * t)) # Square wave bass
    elif style == "Soft Acoustic":
        bg_music = 0.1 * np.sin(2 * np.pi * 330 * t) # Sine wave
    else:
        bg_music = 0.1 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 440 * t))

    # 4. Mix Voice and Background Music
    # We keep the voice loud so the lyrics are clear
    combined = (y_voice * vol_level) + bg_music
    combined = np.clip(combined, -1.0, 1.0)
    
    return combined, sr_rate

# -------- APP FLOW --------

# STEP 1: RECORDING
st.subheader("Step 1: Speak your song idea")
rec = st.audio_input("Record your voice")

if rec:
    # STEP 2: CONVERT TO TEXT
    if 'lyrics' not in st.session_state:
        with st.spinner("Converting your voice to text..."):
            st.session_state.lyrics = get_text_from_voice(rec)
    
    st.success("Voice transcribed successfully!")
    
    # STEP 3: EDIT LYRICS
    st.subheader("Step 2: Refine your Lyrics")
    st.info("The AI heard the words below. You can edit them to make them sound like better lyrics.")
    final_lyrics = st.text_area("Your Lyrics:", st.session_state.lyrics, height=150)
    
    # STEP 4: GENERATE SONG
    if st.button("✨ Generate Final Song with Lyrics"):
        with st.spinner("Generating AI singing voice and music..."):
            song_audio, final_sr = create_final_song(final_lyrics, music_style)
            
            # Save to buffer
            out_buf = io.BytesIO()
            sf.write(out_buf, song_audio, final_sr, format='WAV')
            out_buf.seek(0)
            
            st.divider()
            st.subheader("Step 3: Listen to your Song")
            st.write("You can now hear the words you spoke as an AI song!")
            st.audio(out_buf)
            st.download_button("Download Song", out_buf, "ai_lyrics_song.wav")
