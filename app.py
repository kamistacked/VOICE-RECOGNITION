import streamlit as st
import torch
from speechbrain.inference.speaker import SpeakerRecognition

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SIH Voice Verification", layout="centered", page_icon="🛡️")
st.title("🛡️ Detect-Verify-Prevent")
st.subheader("Deepfake & Speaker Verification Module")
st.divider()

# --- 2. LOAD AI MODEL (CACHED) ---
# st.cache_resource ensures the model only loads once, so the app stays lightning fast
@st.cache_resource
def load_model():
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmpdir_speaker_model"
    )

verifier = load_model()

# --- 3. UI DASHBOARD ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Step 1: Registered Voice")
    st.info("Upload the baseline voice of the authorized user.")
    registered_audio = st.file_uploader("Upload .wav file", type=['wav'])
    if registered_audio:
        st.audio(registered_audio)

with col2:
    st.markdown("### 🎤 Step 2: Live Verification")
    st.info("Record a live sample to test for a match.")
    # This automatically renders a microphone widget
    live_audio = st.audio_input("Record live voice")
    if live_audio:
        st.audio(live_audio)

# --- 4. VERIFICATION LOGIC ---
st.divider()
if st.button("🔐 Authenticate Voice", use_container_width=True):
    if registered_audio and live_audio:
        with st.spinner("Analyzing Voice Biometrics..."):

            # Save the Streamlit memory files to disk temporarily so SpeechBrain can read them
            with open("temp_reg.wav", "wb") as f:
                f.write(registered_audio.getvalue())
            with open("temp_live.wav", "wb") as f:
                f.write(live_audio.getvalue())

            # Compare the two audio files
            score, prediction = verifier.verify_files("temp_reg.wav", "temp_live.wav")

            similarity = score.item()
            is_match = bool(prediction[0])

            # Display Results
            st.markdown(f"**Confidence Score:** `{similarity:.4f}`")

            if is_match:
                st.success("✅ VERIFIED: Speaker match confirmed. Access Granted.")
            else:
                st.error("🚨 ALERT: Voice mismatch! Possible imposter detected.")
    else:
        st.warning("⚠️ Please provide both the registered voice and a live recording first.")