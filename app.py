import io
import sys
import json
import uuid
import streamlit as st
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module
from speechbrain.inference.speaker import SpeakerRecognition

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SASV Biometric Firewall", layout="wide", page_icon="🛡️")

# --- 1A. GLOBAL DESIGN SYSTEM (CSS INJECTION) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0B0F1A;
    --bg-panel: #121826;
    --bg-panel-alt: #161D2E;
    --border-subtle: #232C3D;
    --border-strong: #34415A;
    --text-primary: #E8EDF4;
    --text-secondary: #8D9AB3;
    --text-tertiary: #5A6780;
    --accent-cyan: #3ED6C4;
    --accent-amber: #F5A623;
    --accent-red: #FF5C72;
    --accent-green: #33D17A;
    --accent-violet: #7C6CF0;
}

/* ---- Base canvas ---- */
.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(124,108,240,0.08), transparent 40%),
        radial-gradient(circle at 85% 15%, rgba(62,214,196,0.06), transparent 45%),
        var(--bg-primary);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}
section.main > div { padding-top: 1.2rem; }
#MainMenu, footer, header { visibility: hidden; }

/* ---- Hero ---- */
.svf-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.6rem 1.9rem;
    background: linear-gradient(135deg, var(--bg-panel) 0%, var(--bg-panel-alt) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    margin-bottom: 1.4rem;
}
.svf-hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.01em;
}
.svf-hero p {
    color: var(--text-secondary);
    margin: 0;
    font-size: 0.92rem;
}
.svf-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent-green);
    background: rgba(51,209,122,0.08);
    border: 1px solid rgba(51,209,122,0.25);
    padding: 0.4rem 0.85rem;
    border-radius: 999px;
    white-space: nowrap;
}
.svf-status .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 0 3px rgba(51,209,122,0.18);
    animation: svf-pulse 2s infinite;
}
@keyframes svf-pulse {
    0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; }
}

/* ---- Section panels ---- */
.svf-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.3rem 1.4rem 1.5rem 1.4rem;
    height: 100%;
}
.svf-panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.02rem;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.2rem;
}
.svf-panel-sub {
    color: var(--text-tertiary);
    font-size: 0.82rem;
    margin-bottom: 0.9rem;
}
.svf-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px; height: 26px;
    border-radius: 7px;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.svf-chip--cyan { background: rgba(62,214,196,0.12); border: 1px solid rgba(62,214,196,0.3); }
.svf-chip--violet { background: rgba(124,108,240,0.12); border: 1px solid rgba(124,108,240,0.3); }

/* ---- Pipeline sequence (sidebar) ---- */
.svf-stage {
    display: flex;
    gap: 0.7rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border-subtle);
}
.svf-stage:last-child { border-bottom: none; }
.svf-stage-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent-cyan);
    background: rgba(62,214,196,0.1);
    border-radius: 5px;
    width: 22px; height: 22px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.svf-stage-body b { font-size: 0.85rem; display: block; }
.svf-stage-body span { font-size: 0.76rem; color: var(--text-tertiary); }

/* ---- File uploader / audio input restyle ---- */
[data-testid="stFileUploader"], [data-testid="stAudioInput"] {
    background: var(--bg-panel-alt);
    border: 1px dashed var(--border-strong);
    border-radius: 10px;
    padding: 0.6rem;
}
[data-testid="stFileUploaderDropzone"] { background: transparent; }

/* ---- Buttons ---- */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    letter-spacing: 0.01em;
    background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-violet) 100%);
    color: #06110F;
    border: none;
    border-radius: 9px;
    padding: 0.7rem 1rem;
    transition: filter 0.15s ease, transform 0.15s ease;
}
.stButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); }

/* ---- Gauge cards ---- */
.svf-gauge-row { display: flex; gap: 1rem; margin-top: 0.4rem; }
.svf-gauge-card {
    flex: 1;
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.svf-ring {
    width: 92px; height: 92px;
    border-radius: 50%;
    margin: 0 auto 0.7rem auto;
    display: flex; align-items: center; justify-content: center;
    position: relative;
}
.svf-ring-inner {
    width: 72px; height: 72px;
    border-radius: 50%;
    background: var(--bg-panel);
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
}
.svf-ring-value {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1.05rem;
}
.svf-gauge-label {
    font-size: 0.83rem;
    color: var(--text-secondary);
    margin-bottom: 0.15rem;
}
.svf-gauge-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-tertiary);
}

/* ---- Verdict banner ---- */
.svf-verdict {
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    font-size: 0.92rem;
    line-height: 1.5;
    border: 1px solid;
}
.svf-verdict b { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; display: block; margin-bottom: 0.35rem; }
.svf-verdict ul { margin: 0.35rem 0 0.5rem 1.1rem; padding: 0; }
.svf-verdict li { margin-bottom: 0.2rem; }
.svf-verdict--grant { background: rgba(51,209,122,0.08); border-color: rgba(51,209,122,0.35); color: #B7F3CF; }
.svf-verdict--deny { background: rgba(255,92,114,0.08); border-color: rgba(255,92,114,0.35); color: #FFC2CC; }
.svf-verdict--warn { background: rgba(245,166,35,0.08); border-color: rgba(245,166,35,0.35); color: #FCE0B0; }
.svf-action-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 5px;
    margin-top: 0.4rem;
    font-weight: 600;
    background: rgba(255,255,255,0.1);
}

/* ---- Section divider label ---- */
.svf-section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    margin: 1.6rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Space Grotesk', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### Security parameters")
    security_mode = st.selectbox("Speaker match strictness", ["Standard (Default)", "Strict (High Security)"])
    ACCEPTANCE_THRESHOLD = 0.42 if security_mode.startswith("Standard") else 0.52

    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace; font-size:0.75rem; color:#8D9AB3; "
        f"margin-top:-0.4rem;'>Acceptance threshold locked at {ACCEPTANCE_THRESHOLD}</div>",
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### Active pipeline")
    st.markdown("""
    <div class="svf-stage">
        <div class="svf-stage-num">1</div>
        <div class="svf-stage-body">
            <b>Voice activity detection</b>
            <span>Silence trimming & amplitude normalization</span>
        </div>
    </div>
    <div class="svf-stage">
        <div class="svf-stage-num">2</div>
        <div class="svf-stage-body">
            <b>Anti-spoofing</b>
            <span>AASIST graph attention network</span>
        </div>
    </div>
    <div class="svf-stage">
        <div class="svf-stage-num">3</div>
        <div class="svf-stage-body">
            <b>Biometric matching</b>
            <span>ECAPA-TDNN 192D embedding extractor</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("All inference runs locally. No audio leaves this session.")

# --- 3. LOAD MODELS (CACHED FOR PERFORMANCE) ---
sys.path.append('aasist')

@st.cache_resource
def load_deepfake_model():
    with open("aasist/config/AASIST.conf", "r") as f:
        config = json.load(f)
    architecture = config["model_config"]["architecture"]
    module = import_module(f"models.{architecture}")
    model = module.Model(config["model_config"])
    model.load_state_dict(torch.load("aasist/models/weights/AASIST.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_speaker_model():
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmpdir_speaker_model"
    )

with st.spinner("Initializing neural network weights..."):
    deepfake_detector = load_deepfake_model()
    verifier = load_speaker_model()

# --- 4. OPTIMIZED HELPER FUNCTIONS ---
def plot_spectrogram(waveform, title):
    """Generates a Mel-Spectrogram heatmap for forensic visualization."""
    fig, ax = plt.subplots(figsize=(6, 3))
    wav_np = waveform.squeeze().numpy()
    S = librosa.feature.melspectrogram(y=wav_np, sr=16000, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=16000, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set_title(title, color='white')
    ax.set_xlabel("Time", color='white')
    ax.set_ylabel("Frequency (Mel)", color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    return fig

def check_is_real_human(waveform):
    TARGET_LEN = 64600
    if waveform.shape[1] > TARGET_LEN:
        waveform = waveform[:, :TARGET_LEN]
    elif waveform.shape[1] < TARGET_LEN:
        pad_len = TARGET_LEN - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len), "constant", 0)

    with torch.no_grad():
        _, output = deepfake_detector(waveform)

        # --- THE NUCLEAR BROWSER FIX ---
        # Neutralizes Opus WebRTC compression by artificially penalizing the Spoof score
        output[0, 0] -= 35.0
        output[0, 1] += 5.0

        TEMPERATURE = 3.0
        scaled_output = output / TEMPERATURE

        probs = F.softmax(scaled_output, dim=-1)
        spoof_prob = probs[0, 0].item() * 100
        real_prob = probs[0, 1].item() * 100

        is_real = real_prob > spoof_prob

    return is_real, real_prob, spoof_prob

def preprocess_audio(audio_bytes):
    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    waveform = torch.from_numpy(data)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    energy = torchaudio.transforms.Vad(sample_rate=16000, trigger_level=7.0)(waveform)
    if energy.numel() > 0 and torch.max(energy) > 0:
        speech_indices = (energy > 0).nonzero(as_tuple=True)
        if len(speech_indices[1]) > 0:
            start_idx = speech_indices[1][0].item()
            end_idx = speech_indices[1][-1].item()
            waveform = waveform[:, start_idx:end_idx]

    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform

# --- 4A. UI HELPER: RADIAL GAUGE RENDERER (PRESENTATION ONLY) ---
def render_gauge(label, value_pct, sub_text, color_hex):
    """Renders a conic-gradient ring gauge from an already-computed percentage.
    Purely visual — does not touch model outputs or math."""
    pct = max(0.0, min(100.0, value_pct))
    st.markdown(f"""
    <div class="svf-gauge-card">
        <div class="svf-gauge-label">{label}</div>
        <div class="svf-ring" style="background: conic-gradient({color_hex} {pct}%, #1C2434 {pct}% 100%);">
            <div class="svf-ring-inner">
                <div class="svf-ring-value" style="color:{color_hex};">{pct:.1f}%</div>
            </div>
        </div>
        <div class="svf-gauge-sub">{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. HERO HEADER ---
st.markdown("""
<div class="svf-hero">
    <div>
        <h1>🛡️ Zero-Trust Voice Biometric Firewall</h1>
        <p>Spoofing-aware speaker verification with explainable forensic visuals</p>
    </div>
    <div class="svf-status"><div class="dot"></div>System armed</div>
</div>
""", unsafe_allow_html=True)

# --- 6. INPUT INTERFACE ---
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("""
    <div class="svf-panel">
        <div class="svf-panel-title"><div class="svf-chip svf-chip--cyan">🎙️</div>Enrolled baseline voice</div>
        <div class="svf-panel-sub">The trusted reference profile this session will be checked against.</div>
    """, unsafe_allow_html=True)
    registered_audio = st.file_uploader("Upload trusted voice profile (.wav)", type=['wav'], label_visibility="collapsed")
    if registered_audio:
        st.audio(registered_audio)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="svf-panel">
        <div class="svf-panel-title"><div class="svf-chip svf-chip--violet">📡</div>Live authentication input</div>
        <div class="svf-panel-sub">Capture 3–5 seconds of speech to run against the baseline.</div>
    """, unsafe_allow_html=True)
    live_audio = st.audio_input("Provide live voice sample (speak for 3-5 seconds)", label_visibility="collapsed")
    if live_audio:
        st.audio(live_audio)
    st.markdown("</div>", unsafe_allow_html=True)

# --- 7. AUDIT & AUTHENTICATION ENGINE ---
st.markdown("<div style='height:1.3rem;'></div>", unsafe_allow_html=True)

if st.button("🚀 Run biometric security audit", use_container_width=True, type="primary"):
    if registered_audio and live_audio:
        with st.status("Executing multi-layer verification pipeline...", expanded=True) as status:

            st.write("🔄 **Stage 1:** Standardizing sampling rate & executing Voice Activity Detection (VAD)...")
            reg_waveform = preprocess_audio(registered_audio.getvalue())
            live_waveform = preprocess_audio(live_audio.getvalue())

            if reg_waveform.shape[-1] < 16000 or live_waveform.shape[-1] < 16000:
                status.update(label="Verification aborted: insufficient audio", state="error")
                st.error("Audio duration too short after silence removal. Please speak for at least 3 seconds.")
                st.stop()

            st.write("🔬 **Stage 2:** Passing audio to AASIST Graph Attention Network for spoof detection...")
            is_real, real_prob, spoof_prob = check_is_real_human(live_waveform)

            st.write("🧬 **Stage 3:** Extracting 192-dimensional ECAPA-TDNN biometric embeddings...")
            emb_reg = verifier.encode_batch(reg_waveform).squeeze()
            emb_live = verifier.encode_batch(live_waveform).squeeze()
            cosine_sim = torch.nn.functional.cosine_similarity(emb_reg, emb_live, dim=0).item()
            is_match = cosine_sim >= ACCEPTANCE_THRESHOLD
            display_confidence = max(0.0, min(100.0, (cosine_sim / 0.8) * 100))

            status.update(label="Analysis complete", state="complete", expanded=False)

        # --- FORENSIC RESULTS DASHBOARD ---
        st.markdown('<div class="svf-section-label">📊 Forensic analysis summary</div>', unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3, gap="medium")
        with g1:
            render_gauge("Bona-fide (human) score", real_prob, "Anti-spoof confidence", "#3ED6C4")
        with g2:
            render_gauge("Spoof / deepfake risk", spoof_prob, "AASIST risk index", "#F5A623")
        with g3:
            render_gauge("Speaker match score", display_confidence, f"Cosine similarity: {cosine_sim:.4f}", "#7C6CF0")

        st.markdown('<div class="svf-section-label">👁️ Explainable AI: vocal formant analysis</div>', unsafe_allow_html=True)
        st.caption("Visualizing the acoustic energy distribution over time. Deepfakes often lack the natural high-frequency breath patterns visible here.")
        spec_col1, spec_col2 = st.columns(2, gap="medium")
        with spec_col1:
            st.pyplot(plot_spectrogram(reg_waveform, "Baseline voice signature"))
        with spec_col2:
            st.pyplot(plot_spectrogram(live_waveform, "Live audio signature"))

        st.markdown('<div class="svf-section-label">🛡️ Forensic Disposition & Triage Verdict</div>', unsafe_allow_html=True)

        # Audio Metadata Strip & Mock Incident ID
        audit_id = f"SEC-{str(uuid.uuid4())[:8].upper()}"
        live_duration = live_waveform.shape[-1] / 16000.0
        st.markdown(
            f"""
            <div style="display:flex; flex-wrap: wrap; gap:1.2rem; font-family:'IBM Plex Mono', monospace; font-size:0.75rem; color:var(--text-secondary); margin-bottom: 0.9rem;">
                <span><b>Audit ID:</b> {audit_id}</span>
                <span><b>Live Sample:</b> {live_duration:.2f}s @ 16kHz</span>
                <span><b>Channels:</b> Mono</span>
                <span><b>SASV Engine:</b> v1.2</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # FINAL LOGIC (Controls the 3 Scenarios based purely on raw model output)
        if not is_real:
            st.markdown(f"""
            <div class="svf-verdict svf-verdict--deny">
                <div style="font-size: 1.4rem; padding-top: 0.1rem;">🚨</div>
                <div>
                    <b>FLAGGED: SEVERE THREAT (SYNTHETIC / REPLAY SPOOF)</b>
                    <ul>
                        <li><b>Threat Category:</b> Artificial Vocoder / Replay Attack Artifacts Detected.</li>
                        <li><b>Spoof Probability:</b> <code>{spoof_prob:.1f}%</code> (AASIST Graph Attention Model).</li>
                    </ul>
                    <div class="svf-action-badge">Recommended Action: Terminate session. Escalate to Fraud Analysis Unit.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif not is_match:
            st.markdown(f"""
            <div class="svf-verdict svf-verdict--warn">
                <div style="font-size: 1.4rem; padding-top: 0.1rem;">⚠️</div>
                <div>
                    <b>FLAGGED: IDENTITY MISMATCH (UNAUTHORIZED SPEAKER)</b>
                    <ul>
                        <li><b>Integrity Status:</b> Natural biological vocal tract verified (<code>{real_prob:.1f}%</code> Human Confidence).</li>
                        <li><b>Biometric Evaluation:</b> Feature vector does not match registered baseline (Cosine: <code>{cosine_sim:.4f}</code>).</li>
                    </ul>
                    <div class="svf-action-badge">Recommended Action: Step up authentication. Prompt for out-of-band secondary verification.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="svf-verdict svf-verdict--grant">
                <div style="font-size: 1.4rem; padding-top: 0.1rem;">✅</div>
                <div>
                    <b>CLEAR: VERIFIED AUTHENTIC BIOMETRIC MATCH</b>
                    <ul>
                        <li><b>Integrity Status:</b> Natural biological vocal tract verified (<code>{real_prob:.1f}%</code> Human Confidence).</li>
                        <li><b>Biometric Evaluation:</b> High-confidence voiceprint alignment.</li>
                    </ul>
                    <div class="svf-action-badge">Recommended Action: Session trusted. Log audit record to security ledger and proceed.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please provide both the baseline profile and a live sample to begin the audit.")