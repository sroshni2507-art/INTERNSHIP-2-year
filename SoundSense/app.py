import streamlit as st
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io

# App Configuration
st.set_page_config(page_title="SoundSense", layout="wide", page_icon="🎧")
st.title("🎧 SoundSense – Inclusive AI Music & Movie App")

# -------- SIDEBAR --------
choice = st.sidebar.selectbox(
    "Select Module",
    [
        "Voice → Music",
        "Music Visualizer",
        "Sound Alerts"
    ]
)

# -------- MODULE 1: Voice → Music --------
if choice == "Voice → Music":
    st.header("🎤 Voice to Music")
    st.write("Convert your humming or singing into a synth melody.")
    audio_file = st.file_uploader("Upload your voice (.wav/.mp3)", type=["wav","mp3"])
    
    if audio_file:
        with st.spinner("Processing your voice..."):
            # Load the original audio
            y, sr = librosa.load(audio_file)
            st.text("Original Voice:")
            st.audio(audio_file)

            # 1. Pitch Extraction
            hop_length = 512
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                hop_length=hop_length
            )
            
            # 2. Fix the 0:00 duration issue
            # We must stretch f0 (frames) to match the number of samples in the audio
            f0_cleaned = np.nan_to_num(f0)
            total_samples = len(f0_cleaned) * hop_length
            
            # Linear interpolation to fill gaps between frames
            f0_upsampled = np.interp(
                np.arange(total_samples), 
                np.arange(0, total_samples, hop_length), 
                f0_cleaned
            )

            # 3. Correct Synthesis (Phase Integration)
            # Using cumsum ensures the waveform is continuous and has the correct length
            phase = 2 * np.pi * np.cumsum(f0_upsampled) / sr
            music = np.sin(phase)

            # 4. Normalize and Save
            if np.max(np.abs(music)) > 0:
                music = music / np.max(np.abs(music))
                
                # Use a buffer to avoid file permission issues
                buffer = io.BytesIO()
                sf.write(buffer, music, sr, format='WAV')
                buffer.seek(0)

                st.subheader("🎶 Generated Music Result")
                st.audio(buffer, format='audio/wav')
                st.success("Music generated successfully!")
                
                st.download_button(
                    label="Download Music File",
                    data=buffer,
                    file_name="voice_music.wav",
                    mime="audio/wav"
                )
            else:
                st.error("Could not detect a clear pitch from your voice. Try a clearer recording.")

# -------- MODULE 2: Music Visualizer --------
elif choice == "Music Visualizer":
    st.header("🎶 Music Visualization")
    music_file = st.file_uploader("Upload Music (.wav/.mp3)", type=["wav","mp3"])
    
    if music_file:
        y, sr = librosa.load(music_file)
        
        # Display Audio Player
        st.audio(music_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Beat Tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            st.metric("Tempo (BPM)", round(float(tempo), 2))

            # Waveform Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax, color='blue')
            ax.set_title("Waveform Display")
            st.pyplot(fig)

        with col2:
            # RMS energy (Volume/Intensity)
            energy = np.mean(librosa.feature.rms(y=y))
            
            # Simple Emotion Logic
            if tempo > 120 and energy > 0.05:
                emotion = "Energetic 🔥"
            elif tempo < 80:
                emotion = "Sad/Relaxed 😢"
            else:
                emotion = "Calm 😊"
            
            st.subheader(f"Detected Emotion: {emotion}")
            st.write(f"Intensity Level: {round(energy, 4)}")

# -------- MODULE 4: Sound Alerts --------
elif choice == "Sound Alerts":
    st.header("🚨 Sound Event Alerts")
    st.write("Upload a sound to detect if it is a potential alert (useful for accessibility).")
    sound_file = st.file_uploader("Upload Sound (.wav/.mp3)", type=["wav","mp3"])
    
    if sound_file:
        y, sr = librosa.load(sound_file)
        rms = np.mean(librosa.feature.rms(y=y))

        # 1. Loudness Check
        threshold = 0.05
        if rms > threshold:
            st.error("⚠️ ALERT: Loud Sound Detected!")
        else:
            st.success("✅ Environment is Quiet/Normal")

        # 2. Sound Type Classification (Simplified)
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        st.write("---")
        st.subheader("Analysis Results:")
        
        if spec_centroid < 1500 and zcr < 0.1:
            sound_type = "Low Pitch (Possible Door Knock 🚪)"
        elif spec_centroid > 3000:
            sound_type = "High Pitch (Possible Horn or Alarm 🚗)"
        else:
            sound_type = "Sudden Burst (Possible Explosion/Thump 💥)"

        st.info(f"Classification: {sound_type}")
        st.write(f"Spectral Centroid: {round(spec_centroid, 2)} Hz")
