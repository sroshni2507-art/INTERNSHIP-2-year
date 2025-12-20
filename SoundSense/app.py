import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import speech_recognition as sr
from gtts import gTTS

# App Setup
st.set_page_config(page_title="Voice-to-Lyrics-to-Song", layout="wide", page_icon="🎤")
st.title("🎤 AI Lyricist & Song Generator")
st.write("Flow: Record Voice ➔ Convert to Text ➔ Edit Lyrics ➔ Generate Song")

# -------- SIDEBAR SETTINGS --------
st.sidebar.header("🔊 Volume & Style")
vol_boost = st.sidebar.slider("Volume Gain", 1.0, 10.0, 5.0)
wave_style = st.sidebar.selectbox("Synth Style", ["Square (Loud)", "Sine (Soft)", "Triangle"])

# -------- HELPER FUNCTIONS --------

# 1. Convert Audio to Text (Speech Recognition)
def speech_to_text(audio_bytes):
    r = sr.Recognizer()
    # Convert BytesIO to AudioFile
    audio_data = io.BytesIO(audio_bytes.read())
    with sr.AudioFile(audio_data) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        return text
    except:
        return "Could not understand audio. Please try speaking clearly."

# 2. Convert Voice to Musical Melody (Synthesis)
def synthesize_melody(audio_input, boost=5.0, style="Square (Loud)"):
    y, sample_rate = librosa.load(audio_input, sr=None)
    hop_length = 512
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sample_rate)
    f0_cleaned = np.nan_to_num(f0)
    
    if np.max(f0_cleaned) == 0:
        return None, None

    total_samples = len(f0_cleaned) * hop_length
    f0_upsampled = np.interp(np.arange(total_samples), np.arange(0, total_samples, hop_length), f0_cleaned)

    phase = 2 * np.pi * np.cumsum(f0_upsampled) / sample_rate
    if style == "Square (Loud)":
        music = np.sign(np.sin(phase))
    elif style == "Sine (Soft)":
        music = np.sin(phase)
    else:
        music = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1

    music = np.clip(music * boost, -1.0, 1.0)
    return music, sample_rate

# -------- MAIN APP FLOW --------

# STEP 1: RECORD VOICE
st.subheader("Step 1: Record your message/idea")
recorded_voice = st.audio_input("Speak into the mic")

if recorded_voice:
    # STEP 2: SPEECH TO TEXT
    if 'transcribed_text' not in st.session_state:
        with st.spinner("Converting voice to text..."):
            text = speech_to_text(recorded_voice)
            st.session_state.transcribed_text = text

    st.success("Transcribed Text Found!")
    
    # STEP 3: EDIT LYRICS
    st.subheader("Step 2: Refine your Lyrics")
    lyrics = st.text_area("Edit the text below to make it sound like lyrics:", st.session_state.transcribed_text)
    
    if st.button("Step 3: Generate Song from these Lyrics"):
        with st.spinner("Creating AI Voice and Melody..."):
            # A. Convert refined lyrics to gTTS voice
            tts = gTTS(text=lyrics, lang='en')
            tts_buf = io.BytesIO()
            tts.write_to_fp(tts_buf)
            tts_buf.seek(0)
            
            # B. Convert that AI voice into a Song
            music, sr_out = synthesize_melody(tts_buf, vol_boost, wave_style)
            
            if music is not None:
                final_buf = io.BytesIO()
                sf.write(final_buf, music, sr_out, format='WAV')
                final_buf.seek(0)
                
                st.divider()
                st.subheader("🎉 Final Song Result")
                st.audio(final_buf)
                st.download_button("Download Song", final_buf, "my_ai_lyrics_song.wav")
            else:
                st.error("Melody generation failed. Try adding more words.")

# UPLOAD OPTION (Alternative)
st.divider()
st.subheader("OR Upload a file instead")
uploaded_file = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])
if uploaded_file:
    if st.button("Convert Uploaded File to Song"):
        res_music, res_sr = synthesize_melody(uploaded_file, vol_boost, wave_style)
        if res_music is not None:
            res_buf = io.BytesIO()
            sf.write(res_buf, res_music, res_sr, format='WAV')
            st.audio(res_buf)
