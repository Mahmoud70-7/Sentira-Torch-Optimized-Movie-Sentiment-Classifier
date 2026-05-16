import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# إعدادات الصفحة
st.set_page_config(page_title="Neural Core AI", page_icon="💠", layout="centered")

# --- CSS المفاعل الرقمي ---
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    .portal-overlay {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle, #0a0a1a, #000000);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .activate-btn {
        width: 160px; height: 160px;
        background: #00f2fe;
        border-radius: 50%;
        border: none;
        color: black;
        font-weight: bold;
        font-size: 18px;
        cursor: pointer;
        box-shadow: 0 0 30px #00f2fe;
        animation: pulse-ring 2s infinite;
    }
    @keyframes pulse-ring {
        0% { transform: scale(0.9); box-shadow: 0 0 20px #00f2fe; }
        50% { transform: scale(1); box-shadow: 0 0 50px #4facfe; }
        100% { transform: scale(0.9); box-shadow: 0 0 20px #00f2fe; }
    }
    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 35px;
        border: 1px solid rgba(0, 242, 254, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- منطق الموديل ---
@st.cache_resource
def load_all():
    # تحميل الـ Vectorizer وأوزان الموديل
    vec = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    net = nn.Sequential(
        nn.Linear(5000, 64), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid()
    )
    net.load_state_dict(torch.load('sentiment_model.pth', map_location='cpu'))
    net.eval()
    return vec, net

vectorizer, model = load_all()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean(text):
    cleaned = re.sub(r"[^a-zA-Z0-9]", " ", text).lower().split()
    negation = {'not', 'no', 'never', 'but', 'however'}
    res = [lemmatizer.lemmatize(w) for w in cleaned if w not in stop_words or w in negation]
    return " ".join(res)

# --- نظام التشغيل عن طريق الرابط (لحل مشاكل التحميل) ---
if 'activated' not in st.session_state:
    st.session_state.activated = False

query_params = st.query_params
if "action" in query_params and query_params["action"] == "activate":
    st.session_state.activated = True

if not st.session_state.activated:
    st.markdown(f"""
        <div class="portal-overlay">
            <div style="margin-bottom: 40px; font-size: 24px; letter-spacing: 5px; color:#00f2fe;">SYSTEM OFFLINE</div>
            <a href="/?action=activate" target="_self">
                <button class="activate-btn">ACTIVATE<br>CORE</button>
            </a>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# الواجهة الرئيسية (تظهر بعد التفعيل)
if st.session_state.activated:
    st.markdown('<h1 style="text-align:center; color:#00f2fe;">NEURAL CORE AI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    user_input = st.text_area("✍️ Input Stream:", placeholder="System ready for analysis...")

    if st.button("EXECUTE ANALYSIS"):
        if user_input.strip():
            proc = clean(user_input)
            tensor_in = torch.tensor(vectorizer.transform([proc]).toarray(), dtype=torch.float32)
            with torch.no_grad():
                prob = model(tensor_in).item()

            st.markdown("---")
            if prob >= 0.5:
                st.success(f"🌟 ANALYSIS: POSITIVE ({prob*100:.1f}%)")
                st.balloons()
            else:
                st.error(f"😞 ANALYSIS: NEGATIVE ({(1-prob)*100:.1f}%)")

    if st.button("TERMINATE SESSION"):
        st.query_params.clear()
        st.session_state.activated = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
