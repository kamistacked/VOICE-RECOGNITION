import io
import streamlit as st
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SIH Voice Verification", layout="centered", page_icon="🛡️")
st.title("🛡️ Detect-Verify-Prevent")
st.subheader("Deepfake & Speaker Verification Module")
st.divider()

# --- 2. LOAD AI MODEL (CACHED) ---
@st.cache_resource
def load_model():
    # Downloads and loads the pre-trained ECAPA-TDNN model
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmpdir_speaker_model"
    )

verifier = load_model()

# --- 3. AUDIO PROCESSING LOGIC ---
def preprocess_audio(audio_bytes):
    """
    Reads raw audio bytes, downmixes to mono, resamples to 16kHz,
    trims silence, and normalizes volume.
    """
    # 1. Read audio bytes directly using soundfile (avoids torchcodec error)
    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')

    # 2. Convert NumPy array to PyTorch tensor
    waveform = torch.from_numpy(data)

    # 3. Handle channels (make it Mono)
    if waveform.ndim == 1:
        # [num_samples] -> [1, num_samples]
        waveform = waveform.unsqueeze(0)
    else:
        # [num_samples, channels] -> [channels, num_samples]
        waveform = waveform.t()
        # Downmix stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 4. Resample to 16,000 Hz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # 5. Trim quiet/silent frames (VAD energy thresholding)
    energy = torchaudio.transforms.Vad(sample_rate=16000, trigger_level=7.0)(waveform)
    if energy.numel() > 0 and torch.max(energy) > 0:
        speech_indices = (energy > 0).nonzero(as_tuple=True)
        if len(speech_indices[1]) > 0:
            start_idx = speech_indices[1][0].item()
            end_idx = speech_indices[1][-1].item()
            waveform = waveform[:, start_idx:end_idx]

    # 6. Volume peak normalization
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    return waveform

# --- 4. UI DASHBOARD ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Baseline Voice")
    st.info("Upload the baseline voice of the authorized user.")
    registered_audio = st.file_uploader("Upload reference audio (.wav)", type=['wav'])
    if registered_audio:
        st.audio(registered_audio)

with col2:
    st.markdown("### 🎤 Live Verification")
    st.info("Record a live sample to test (speak for 3-5 seconds).")
    live_audio = st.audio_input("Record live voice")
    if live_audio:
        st.audio(live_audio)

# --- 5. VERIFICATION LOGIC ---
st.divider()
if st.button("🔐 Authenticate Voice", use_container_width=True):
    if registered_audio and live_audio:
        with st.spinner("Analyzing biometric feature vectors..."):

            # Clean and prepare both audio sources
            reg_waveform = preprocess_audio(registered_audio.getvalue())
            live_waveform = preprocess_audio(live_audio.getvalue())

            # Ensure minimum speech length (reject files that are just 1 second of silence)
            if reg_waveform.shape[-1] < 16000 or live_waveform.shape[-1] < 16000:
                st.warning("⚠️ Utterance is too short after trimming silence. Please speak clearly for at least 3 seconds.")
            else:
                # Extract 192-dimensional ECAPA embeddings
                emb_reg = verifier.encode_batch(reg_waveform).squeeze()
                emb_live = verifier.encode_batch(live_waveform).squeeze()

                # Calculate standard Cosine Similarity (-1.0 to +1.0)
                cosine_sim = torch.nn.functional.cosine_similarity(emb_reg, emb_live, dim=0).item()

                # Strictly calibrated threshold for distinct speakers
                # (Same speakers > ~0.40, Different speakers < ~0.25)
                ACCEPTANCE_THRESHOLD = 0.42
                is_match = cosine_sim >= ACCEPTANCE_THRESHOLD

                # Convert cosine similarity into an intuitive display percentage
                display_confidence = max(0.0, min(100.0, (cosine_sim / 0.8) * 100))

                # Display Results
                col_res1, col_res2 = st.columns(2)
                col_res1.metric(
                    label="Cosine Similarity",
                    value=f"{cosine_sim:.4f}",
                    help="Raw similarity score strictly bounded between -1.0 and +1.0"
                )
                col_res2.metric(
                    label="Normalized Confidence",
                    value=f"{display_confidence:.1f}%"
                )

                if is_match:
                    st.success("✅ **VERIFIED:** Voice biometric confirmed. Access granted.")
                else:
                    st.error("🚨 **ALERT: MISMATCH!** Unauthorized speaker or imposter detected.")
    else:
        st.warning("⚠️ Please provide both the registered voice and a live test recording.")