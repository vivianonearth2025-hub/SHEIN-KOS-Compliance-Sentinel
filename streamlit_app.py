import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import numpy as np
import tempfile
import os

# ============================================================
# CONFIG
# ============================================================
PIPELINE_A_MODEL = "Vivianonearth666/SHEIN_compliance_text_v2"
PIPELINE_C_MODEL = "JLi09/shein_compliance_ATT_Sentiment"
PIPELINE_B_MODEL = "openai/whisper-small"

HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SHEIN KOS Compliance Sentinel",
    page_icon="🛡️",
    layout="wide",
)

# ============================================================
# MODEL LOADERS (cached after first load)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_pipeline_a():
    model = AutoModelForSequenceClassification.from_pretrained(PIPELINE_A_MODEL, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(PIPELINE_A_MODEL, token=HF_TOKEN)
    model.eval()
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_pipeline_b():
    return pipeline("automatic-speech-recognition", model=PIPELINE_B_MODEL, token=HF_TOKEN)

@st.cache_resource(show_spinner=False)
def load_pipeline_c():
    model = AutoModelForSequenceClassification.from_pretrained(PIPELINE_C_MODEL, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(PIPELINE_C_MODEL, token=HF_TOKEN)
    model.eval()
    return model, tokenizer


# ============================================================
# HELPERS
# ============================================================
def preprocess_twitter(text):
    new_text = []
    for t in str(text).split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def predict_compliance(text, model, tokenizer, use_twitter_preprocessing=True):
    """returns (label, confidence, all_scores_dict)"""
    processed = preprocess_twitter(text) if use_twitter_preprocessing else text
    inputs = tokenizer(processed, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
    pred_id = int(np.argmax(probs))
    return (
        model.config.id2label[pred_id],
        float(probs[pred_id]),
        {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}
    )


ACTION_MAP = {
    'compliant':                 ('✅ Approve', 'success'),
    'ftc_disclosure_violation':  ('🚫 Block — request clear disclosure (e.g. #ad)', 'error'),
    'unsubstantiated_claims':    ('⚠️ Flag for review — verify supporting evidence', 'warning'),
    'greenwashing':              ('⚠️ Flag for review — verify environmental claims', 'warning'),
}


def display_result(pred_label, confidence, all_scores):
    action_text, severity = ACTION_MAP.get(pred_label, ('Unknown', 'warning'))
    col1, col2 = st.columns([3, 1])
    with col1:
        msg = f"**Predicted class:** `{pred_label}`"
        if severity == 'success':   st.success(msg)
        elif severity == 'error':   st.error(msg)
        else:                        st.warning(msg)
    with col2:
        st.metric("Confidence", f"{confidence*100:.1f}%")
    
    st.markdown(f"**Suggested action:** {action_text}")
    st.markdown("##### All class probabilities")
    for label, score in sorted(all_scores.items(), key=lambda x: -x[1]):
        st.write(f"**{label}**")
        st.progress(score, text=f"{score*100:.1f}%")


# ============================================================
# HEADER
# ============================================================
st.title("🛡️ SHEIN KOS Compliance Sentinel")
st.markdown("AI-powered compliance screening for cross-border influencer marketing. Detects FTC, ASA, and FDA-style violations in written posts and live-stream audio.")

if not HF_TOKEN:
    st.error("⚠️ HF_TOKEN secret not configured.")
    st.stop()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📝 Pre-Screening", "🎤 Live Monitoring", "ℹ️ About"])

# ---------- Tab 1 ----------
with tab1:
    st.header("Pre-Screen Influencer Post")
    st.markdown("Paste a written SHEIN influencer post to classify for FTC disclosure, unsubstantiated claims, or greenwashing violations.")
    
    text_input = st.text_area("Influencer post text:", height=150, placeholder="e.g., Y'all need to check out this SHEIN haul use code MAYA15 for 15% off!!")
    
    if st.button("🔍 Analyze", type="primary", key="analyze_text"):
        if not text_input.strip():
            st.warning("Please paste some text to analyze.")
        else:
            with st.spinner("Loading model (first time may take 1-2 min)…"):
                model, tokenizer = load_pipeline_a()
            with st.spinner("Analyzing..."):
                pred_label, confidence, all_scores = predict_compliance(text_input, model, tokenizer)
            st.markdown("---")
            display_result(pred_label, confidence, all_scores)

# ---------- Tab 2 ----------
with tab2:
    st.header("Monitor Live Stream Audio")
    st.markdown("Upload an audio clip from a SHEIN live haul. Whisper-small transcribes the speech, then Pipeline C (fine-tuned on spoken-style data) classifies the transcript.")
    
    audio_file = st.file_uploader("Choose audio file (MP3, WAV, M4A):", type=['mp3', 'wav', 'm4a', 'ogg'])
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("🎤 Transcribe & Analyze", type="primary", key="analyze_audio"):
            suffix = os.path.splitext(audio_file.name)[1] or '.mp3'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.getbuffer())
                tmp_path = tmp.name
            
            try:
                with st.spinner("Loading Whisper (first time may take 2-3 min)…"):
                    whisper_pipe = load_pipeline_b()
                with st.spinner("🎙️ Transcribing audio…"):
                    transcript = whisper_pipe(tmp_path)['text'].strip()
                
                st.markdown("### 📝 Whisper Transcription")
                st.info(transcript if transcript else "(no speech detected)")
                
                if transcript:
                    with st.spinner("Loading Pipeline C…"):
                        model_c, tokenizer_c = load_pipeline_c()
                    with st.spinner("🎯 Classifying compliance…"):
                        pred_label, confidence, all_scores = predict_compliance(
                            transcript, model_c, tokenizer_c, use_twitter_preprocessing=False
                        )
                    st.markdown("### 🎯 Pipeline C Classification")
                    display_result(pred_label, confidence, all_scores)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# ---------- Tab 3: About ----------
with tab3:
    st.header("About This Project")
    st.markdown("**SHEIN KOS Compliance Sentinel** is an AI-powered compliance risk screening system for SHEIN's cross-border influencer marketing.")
    st.markdown("Built as a capstone project for **HKUST ISOM5240 — Deep Learning Business Applications with Python** (Spring 2026) by Jasper (Jiayi) Li and Vivian (Wei) Wu.")
    
    st.markdown("---")
    st.subheader("🏗️ System Architecture")
    st.markdown("- **Pipeline A** — Written post classifier, fine-tuned `cardiffnlp/twitter-roberta-base-2022-154m`")
    st.markdown("- **Pipeline B** — Speech-to-text, `openai/whisper-small`")
    st.markdown("- **Pipeline C** — Spoken transcript classifier, fine-tuned `distilbert-base-uncased`")
    
    st.markdown("---")
    st.subheader("📊 Pipeline A Final Performance (V2)")
    st.markdown("- Test Accuracy: **96.97%** (V1: 95.45%)")
    st.markdown("- F1 Macro: **97.05%** (V1: 95.48%)")
    st.markdown("- FTC recall: 0.8750 → **0.9375** via hard negative mining")
    
    st.markdown("---")
    st.subheader("📊 Pipeline B WER")
    st.markdown("- whisper-tiny: **18.18%**")
    st.markdown("- whisper-base: **10.39%**")
    st.markdown("- **whisper-small** (selected): **7.79%**")
    
    st.markdown("---")
    st.subheader("📊 Pipeline C")
    st.markdown("- Test Accuracy: **88.3%**")
    st.markdown("- F1 Macro: **89.06%**")
    
    st.markdown("---")
    st.subheader("🔗 Resources")
    st.markdown(f"- Pipeline A V2: [{PIPELINE_A_MODEL}](https://huggingface.co/{PIPELINE_A_MODEL})")
    st.markdown(f"- Pipeline C: [{PIPELINE_C_MODEL}](https://huggingface.co/{PIPELINE_C_MODEL})")
    st.markdown(f"- Pipeline B: [{PIPELINE_B_MODEL}](https://huggingface.co/{PIPELINE_B_MODEL})")
    
    st.markdown("---")
    st.caption("Authors: Jasper (Jiayi) Li, Vivian (Wei) Wu. Disclaimer: Academic prototype — outputs are advisory only.")
