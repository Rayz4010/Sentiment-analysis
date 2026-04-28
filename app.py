import streamlit as st
import numpy as np
import re

# ── Backend (kept internal) ───────────────────────────────────────────────────
def _load_backend():
    import spacy
    return spacy

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; border-right: 1px solid #2d3148; }
    .card { background: #1a1d27; border: 1px solid #2d3148; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .card-highlight { background: linear-gradient(135deg, #1a1d27 0%, #1e2235 100%); border: 1px solid #3d4266; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .badge-hate { display: inline-block; background: linear-gradient(135deg, #ff4757, #c0392b); color: white; font-weight: 700; font-size: 1.1rem; padding: 0.5rem 1.4rem; border-radius: 30px; letter-spacing: 1px; }
    .badge-offensive { display: inline-block; background: linear-gradient(135deg, #ffa502, #e67e22); color: white; font-weight: 700; font-size: 1.1rem; padding: 0.5rem 1.4rem; border-radius: 30px; letter-spacing: 1px; }
    .badge-neither { display: inline-block; background: linear-gradient(135deg, #2ed573, #27ae60); color: white; font-weight: 700; font-size: 1.1rem; padding: 0.5rem 1.4rem; border-radius: 30px; letter-spacing: 1px; }
    .conf-label { font-size: 0.85rem; color: #a0a3b1; margin-bottom: 2px; }
    .metric-tile { background: #1a1d27; border: 1px solid #2d3148; border-radius: 10px; padding: 1rem; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #7c83fd; }
    .metric-label { font-size: 0.8rem; color: #a0a3b1; margin-top: 4px; }
    .hist-item { background: #1a1d27; border-left: 4px solid #3d4266; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; font-size: 0.9rem; }
    h1, h2, h3 { color: #e8e9f3 !important; }
    p, label { color: #c5c7d4 !important; }
    .stTextArea textarea { background: #1a1d27 !important; color: #e8e9f3 !important; border: 1px solid #3d4266 !important; border-radius: 8px !important; }
    .stButton > button { background: linear-gradient(135deg, #7c83fd, #5c6bc0); color: white; border: none; border-radius: 8px; font-weight: 600; width: 100%; padding: 0.6rem 0; transition: opacity 0.2s; }
    .stButton > button:hover { opacity: 0.88; }
    .stProgress > div > div { background: linear-gradient(90deg, #7c83fd, #c471ed) !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SENTENCE_LENGTH = 20
LABELS      = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
BADGE_CLASS = {0: "badge-hate", 1: "badge-offensive", 2: "badge-neither"}
EMOJI       = {0: "🚨", 1: "⚠️", 2: "✅"}
COLOR       = {0: "#ff4757", 1: "#ffa502", 2: "#2ed573"}

# ── Load everything once ──────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    spacy = _load_backend()
    nlp   = spacy.load("en_core_web_sm")
    model = tf.keras.models.load_model("hate_speech.keras")

    with open("tokenizer.json", "r") as f:
        tokenizer = tokenizer_from_json(f.read())

    return model, nlp, tokenizer, pad_sequences

# ── Classify ──────────────────────────────────────────────────────────────────
def classify(text: str):
    model, nlp, tokenizer, pad_sequences = load_resources()

    # Mirror training preprocessing exactly
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(t.lemma_ for t in nlp(text))
    text = " ".join(t.text for t in nlp(text) if not t.is_stop)

    # Use the saved tokenizer — same vocab as training
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, padding="pre", maxlen=SENTENCE_LENGTH)

    probs = model.predict(np.array(padded), verbose=0)[0]
    return int(np.argmax(probs)), probs

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "counts" not in st.session_state:
    st.session_state.counts = {0: 0, 1: 0, 2: 0}

# ── Boot ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    try:
        load_resources()
        model_ok = True
    except FileNotFoundError as e:
        model_ok = False
        missing = str(e)
        load_err = missing
    except Exception as e:
        model_ok = False
        load_err = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##Hate Speech Detector")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. **Clean** – strips non-alpha characters  
2. **Lemmatise** – reduces words to root form  
3. **Stop-word removal** – drops filler words  
4. **Tokenize** – words → consistent vocab IDs  
5. **Padding** – fixed-length sequences  
6. **LSTM RNN** – predicts category  
""")
    st.markdown("---")
    st.markdown("### Categories")
    st.markdown("🚨 **Hate Speech** – targeted hatred")
    st.markdown("⚠️ **Offensive Language** – profane but not hateful")
    st.markdown("✅ **Neither** – clean or neutral")
    st.markdown("---")
    st.markdown("### Session Stats")
    total = sum(st.session_state.counts.values())
    for cls, label in LABELS.items():
        cnt = st.session_state.counts[cls]
        pct = (cnt / total * 100) if total else 0
        st.markdown(f"{EMOJI[cls]} **{label}**: {cnt} ({pct:.0f}%)")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.counts  = {0: 0, 1: 0, 2: 0}
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ Hate Speech Detection")
st.markdown("Paste any text below and the LSTM model will classify it in real-time.")

if not model_ok:
    st.error(f"❌ Failed to load: `{load_err}`")
    if "tokenizer.json" in load_err:
        st.markdown("""
### Missing `tokenizer.json` — run this in Colab to generate it:
```python
from tensorflow.keras.preprocessing.text import Tokenizer
import json

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['final_tweet'])  # your preprocessed column

with open('tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())
```
Download `tokenizer.json` and place it in the same folder as `app.py`.
""")
    st.stop()

col_input, col_result = st.columns([3, 2], gap="large")

with col_input:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if "_ex" not in st.session_state:
        st.session_state["_ex"] = ""

    user_text = st.text_area(
        "Enter text to analyse",
        value=st.session_state["_ex"],
        placeholder="Type or paste a tweet / sentence here…",
        height=160,
        label_visibility="collapsed",
    )
    c1, c2 = st.columns(2)
    analyse_btn = c1.button("🔍 Analyse", use_container_width=True)
    if c2.button("📋 Try Example", use_container_width=True):
        import random
        examples = [
            "You people are disgusting and don't deserve to live here.",
            "This damn weather is really screwing up my plans today.",
            "The weather today is beautiful and I feel great!",
        ]
        st.session_state["_ex"] = random.choice(examples)
        st.rerun()
    wc = len(user_text.split()) if user_text.strip() else 0
    st.caption(f"Words: {wc} | Characters: {len(user_text)}")
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="card-highlight">', unsafe_allow_html=True)
    if analyse_btn and user_text.strip():
        pred, probs = classify(user_text)
        st.session_state.counts[pred] += 1
        st.session_state.history.insert(0, {
            "text": user_text[:120] + ("…" if len(user_text) > 120 else ""),
            "pred": pred,
            "probs": probs.tolist(),
        })
        st.markdown(f"### {EMOJI[pred]} Result")
        st.markdown(f'<span class="{BADGE_CLASS[pred]}">{LABELS[pred].upper()}</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Confidence scores**")
        for cls in range(3):
            pct = float(probs[cls]) * 100
            st.markdown(f'<div class="conf-label">{EMOJI[cls]} {LABELS[cls]}: {pct:.1f}%</div>', unsafe_allow_html=True)
            st.progress(float(probs[cls]))
        st.markdown("---")
        if pred == 0:
            st.error("⚠️ This text contains **hate speech** targeting individuals or groups.")
        elif pred == 1:
            st.warning("ℹ️ This text uses **offensive language** but may not constitute hate speech.")
        else:
            st.success("✔️ This text appears **neutral or inoffensive**.")
    elif analyse_btn:
        st.info("Please enter some text first.")
    else:
        st.markdown("### 👆 Enter text and click Analyse")
        st.markdown("Results and confidence scores will appear here.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
total = sum(st.session_state.counts.values())
if total:
    st.markdown("---")
    st.markdown("### 📊 Session Overview")
    m1, m2, m3, m4 = st.columns(4)
    tiles = [(total, "Total Analysed"), (st.session_state.counts[0], "🚨 Hate Speech"),
             (st.session_state.counts[1], "⚠️ Offensive"), (st.session_state.counts[2], "✅ Clean")]
    for col, (val, lbl) in zip([m1, m2, m3, m4], tiles):
        col.markdown(f'<div class="metric-tile"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕓 Analysis History")
    for item in st.session_state.history[:10]:
        pred = item["pred"]
        conf = max(item["probs"]) * 100
        st.markdown(
            f'<div class="hist-item">'
            f'<span style="color:{COLOR[pred]};font-weight:700;">{EMOJI[pred]} {LABELS[pred]}</span>'
            f' <span style="color:#666;font-size:0.8rem;">({conf:.0f}% confidence)</span><br>'
            f'<span style="color:#a0a3b1;">{item["text"]}</span></div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p style="text-align:center;color:#555;font-size:0.8rem;">Powered by an LSTM-based RNN trained on Twitter data &nbsp;|&nbsp; ~90% accuracy &nbsp;|&nbsp; 3-class classification</p>', unsafe_allow_html=True)