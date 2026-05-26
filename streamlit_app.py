import streamlit as st
import requests
import time

# ============================================================
# CONFIG 
# ============================================================
PIPELINE_A_MODEL = "Vivianonearth666/SHEIN_compliance_text_v2"          
PIPELINE_C_MODEL = "Vivianonearth666/shein_compliance_ATT_sentiment"   
PIPELINE_B_MODEL = "openai/whisper-small"

# HF token from Streamlit Cloud secrets
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
# HF INFERENCE API HELPERS
# ============================================================
def preprocess_twitter(text):
    """cardiffnlp models need @user and http placeholders"""
    new_text = []
    for t in str(text).split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def call_hf_text(text, model_id, max_retries=3):
    """Call HF Inference API for text classification with cold-start retry"""
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    for attempt in range(max_retries):
        r = requests.post(url, headers=headers, json={"inputs": text})
        
        if r.status_code == 503:
            wait = r.json().get('estimated_time', 20)
            st.info(f"Model warming up (try {attempt+1}/{max_retries})… ~{wait:.0f}s")
            time.sleep(wait)
            continue
        
        r.raise_for_status()
        return r.json()
    
    raise RuntimeError(f"Model {model_id} still loading after {max_retries} retries")


def call_hf_audio(audio_bytes, model_id, max_retries=3):
    """Call HF Inference API for Whisper ASR"""
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    for attempt in range(max_retries):
        r = requests.post(url, headers=headers, data=audio_bytes)
        
        if r.status_code == 503:
            wait = r.json().get('estimated_time', 30)
            st.info(f"Whisper warming up (try {attempt+1}/{max_retries})… ~{wait:.0f}s")
            time.sleep(wait)
            continue
        
        r.raise_for_status()
        return r.json()
    
    raise RuntimeError(f"Whisper still loading after {max_retries} retries")


def parse_classification(api_response):
    """HF API returns [[{label, score}, ...]] sorted by score desc"""
    scores_list = api_response[0] if isinstance(api_response[0], list) else api_response
    all_scores = {item['label']: item['score'] for item in scores_list}
    top = max(scores_list, key=lambda x: x['score'])
    return top['label'], top['score'], all_scores


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
# HEADER + TOKEN CHECK
# ============================================================
st.title("🛡️ SHEIN KOS Compliance Sentinel")
st.markdown("AI-powered compliance screening for cross-border influencer marketing. Detects FTC, ASA, and FDA-style violations in written posts and live-stream audio.")

if not HF_TOKEN:
    st.error("⚠️ HF_TOKEN secret not configured. Go to Streamlit Cloud → app Settings → Secrets and add HF_TOKEN = \"hf_xxx\"")
    st.stop()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📝 Pre-Screening", "🎤 Live Monitoring", "ℹ️ About"])

# ---------- Tab 1: Pipeline A ----------
with tab1:
    st.header("Pre-Screen Influencer Post")
    st.markdown("Paste a written SHEIN influencer post to classify for FTC disclosure, unsubstantiated claims, or greenwashing violations.")
    
    text_input = st.text_area("Influencer post text:", height=150, placeholder="e.g., Y'all need to check out this SHEIN haul use code MAYA15 for 15% off!!")
    
    if st.button("🔍 Analyze", type="primary", key="analyze_text"):
        if not text_input.strip():
            st.warning("Please paste some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    processed = preprocess_twitter(text_input)   # cardiffnlp preprocessing
                    api_result = call_hf_text(processed, PIPELINE_A_MODEL)
                    pred_label, confidence, all_scores = parse_classification(api_result)
                    st.markdown("---")
                    display_result(pred_label, confidence, all_scores)
                except Exception as e:
                    st.error(f"Error: {e}")

# ---------- Tab 2: Pipeline B + C ----------
with tab2:
    st.header("Monitor Live Stream Audio")
    st.markdown("Upload an audio clip from a SHEIN live haul. The system transcribes the speech (Whisper-small) and classifies the transcript for compliance violations.")
    
    audio_file = st.file_uploader("Choose audio file (MP3, WAV, M4A):", type=['mp3', 'wav', 'm4a', 'ogg'])
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("🎤 Transcribe & Analyze", type="primary", key="analyze_audio"):
            try:
                audio_bytes = audio_file.read()
                
                with st.spinner("🎙️ Transcribing audio with Whisper..."):
                    asr_result = call_hf_audio(audio_bytes, PIPELINE_B_MODEL)
                    transcript = asr_result.get('text', '').strip()
                
                st.markdown("### 📝 Whisper Transcription")
                st.info(transcript if transcript else "(no speech detected)")
                
                if transcript:
                    with st.spinner("🎯 Classifying compliance (Pipeline C)..."):
                        # distilbert — no Twitter preprocessing
                        api_result = call_hf_text(transcript, PIPELINE_C_MODEL)
                        pred_label, confidence, all_scores = parse_classification(api_result)
                    
                    st.markdown("### 🎯 Pipeline C Classification")
                    display_result(pred_label, confidence, all_scores)
            except Exception as e:
                st.error(f"Error: {e}")

# ---------- Tab 3: About ----------
with tab3:
    st.header("About This Project")
    st.markdown("**SHEIN KOS Compliance Sentinel** is an AI-powered compliance risk screening system for SHEIN's cross-border influencer marketing.")
    st.markdown("Built as a capstone project for **HKUST ISOM5240 — Deep Learning Business Applications with Python** (Spring 2026) by Jasper (Jiayi) Li and Vivian.")
    
    st.markdown("---")
    st.subheader("🏗️ System Architecture")
    st.markdown("- **Pipeline A** — Written post classifier, fine-tuned `cardiffnlp/twitter-roberta-base-2022-154m`")
    st.markdown("- **Pipeline B** — Speech-to-text, `openai/whisper-small`")
    st.markdown("- **Pipeline C** — Spoken transcript classifier, fine-tuned `distilbert-base-uncased`")
    st.markdown("Models are invoked via Hugging Face Inference API to keep the Streamlit Cloud footprint minimal.")
    
    st.markdown("---")
    st.subheader("📊 Pipeline A Final Performance (V2)")
    st.markdown("- Test Accuracy: **96.97%** (V1: 95.45%)")
    st.markdown("- F1 Macro: **97.05%** (V1: 95.48%)")
    st.markdown("- FTC recall: 0.8750 → **0.9375** via hard negative mining")
    
    st.markdown("---")
    st.subheader("📊 Pipeline B WER")
    st.markdown("- whisper-tiny: **15.19%**")
    st.markdown("- whisper-base: **7.59%**")
    st.markdown("- **whisper-small** (selected): **5.06%**")
    
    st.markdown("---")
    st.subheader("📊 Pipeline C")
    st.markdown("- Test Accuracy: **88.30%**")
    st.markdown("- F1 Macro: **89.07%**")
    
    st.markdown("---")
    st.subheader("🔗 Resources")
    st.markdown(f"- Pipeline A V2: [{PIPELINE_A_MODEL}](https://huggingface.co/{PIPELINE_A_MODEL})")
    st.markdown(f"- Pipeline C: [{PIPELINE_C_MODEL}](https://huggingface.co/{PIPELINE_C_MODEL})")
    st.markdown(f"- Pipeline B: [{PIPELINE_B_MODEL}](https://huggingface.co/{PIPELINE_B_MODEL})")
    
    st.markdown("---")
    st.caption("Authors: Jasper (Jiayi) Li, Vivian. Disclaimer: Academic prototype — outputs are advisory only.")
