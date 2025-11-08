# smart_clinical_extractor_app.py
# ================================
# 🧠 Smart Clinical Extractor
# Professional, section-wise UI for extracting clinical conditions
# ================================

import streamlit as st
import pandas as pd
import re, io, os, warnings
from pathlib import Path
from typing import List, Dict, Set
import spacy
from rapidfuzz import process, fuzz

# ---------------------- Streamlit & Environment Setup ----------------------
st.set_page_config(page_title="🧠 Smart Clinical Extractor", layout="wide")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TORCH_LOGS"] = "none"

# ---------------------- Hero Section ----------------------
st.markdown("""
<div style="text-align:center; padding:50px 20px;">
    <h1 style="
        font-size:3rem; 
        font-weight:800; 
        color:#4f46e5; 
        text-shadow: 0 0 10px #4f46e5, 0 0 20px #4338ca;
        animation: glowText 2s infinite alternate;
        margin-bottom:15px;
    ">🧠 Smart Clinical Extractor</h1>
    <p style="
        font-size:1.1rem; 
        color:#a0a0a0;
        max-width:650px;
        margin:0 auto;
        line-height:1.5;
    ">
        Paste your clinical notes or load a file — the extractor automatically detects diseases, conditions, and key clinical terms.
    </p>
</div>
<style>
@keyframes glowText {
  0% { text-shadow: 0 0 10px #4f46e5, 0 0 20px #4338ca; }
  50% { text-shadow: 0 0 15px #4338ca, 0 0 30px #4f46e5; }
  100% { text-shadow: 0 0 10px #4f46e5, 0 0 20px #4338ca; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Global Styles ----------------------
st.markdown("""
<style>
body { background-color:#0f1117; color:#e0e0e0; font-family:'Inter',sans-serif; overflow-x:hidden; }
h1,h2,h3 { font-weight:700; color:#fff; margin-bottom:0.5rem; }
.container, .card, form, .stForm { background-color:#1c1c1c; border-radius:12px; padding:25px 30px; margin:20px auto; border:1px solid #2a2a2a; box-shadow:0 4px 12px rgba(0,0,0,0.5); transition: all 0.3s ease-in-out; }
.container:hover, .card:hover { border-color:#4f46e5; box-shadow:0 6px 18px rgba(79,70,229,0.4); transform:translateY(-2px); }
.stButton>button { background-color:#4f46e5; color:white; border:none; padding:12px 25px; font-weight:600; border-radius:6px; cursor:pointer; transition: all 0.2s ease; }
.stButton>button:hover { background-color:#4338ca; transform:translateY(-2px) scale(1.05); box-shadow:0 0 10px #4f46e5; }
.stTextArea textarea { border-radius:6px !important; border:1px solid #2c2c2c !important; background-color:#1c1c1c !important; color:#e0e0e0 !important; font-family:'Inter',sans-serif !important; font-size:0.95rem !important; transition:border 0.2s ease, box-shadow 0.2s ease; }
.stTextArea textarea:focus { border-color:#4f46e5 !important; box-shadow:0 0 10px #4f46e5; }
.result-chip { background-color:#4f46e5; color:#fff; padding:6px 14px; margin:4px; border-radius:8px; font-weight:600; display:inline-block; animation: glowPulse 2s infinite alternate; }
.result-chip:hover { background-color:#4338ca; transform:translateY(-1px); }
@keyframes glowPulse { 0% { box-shadow:0 0 5px #4f46e5; } 50% { box-shadow:0 0 12px #4338ca; } 100% { box-shadow:0 0 5px #4f46e5; } }
footer, header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------- Config ----------------------
DISEASE_DICT_PATH = "diseases_cleaned_final.txt"
USE_SCISPACY = False
SCISPACY_MODEL = "en_core_sci_sm"
SPACY_MODEL = "en_core_web_sm"
FUZZ_THRESHOLD = {"exact":100,"strict":95,"clinical":88,"smart":85,"lenient":75}
STOPWORDS = set(["a","of","the","and","for","with","in","on","is","are","by","to","from","patient","history"])

# ---------------------- Load Models & Data ----------------------
@st.cache_resource
def load_nlp_model():
    model = SCISPACY_MODEL if USE_SCISPACY else SPACY_MODEL
    try:
        return spacy.load(model, disable=["parser","tagger"])
    except Exception as e:
        st.error(f"❌ Failed to load NLP model: {e}")
        st.stop()

@st.cache_data
def load_disease_dict(path:str) -> List[str]:
    p=Path(path)
    if not p.exists(): st.error(f"❌ Disease dictionary not found at {path}"); st.stop()
    with p.open("r", encoding="utf-8") as f: items=[line.strip().lower() for line in f if line.strip()]
    return sorted(set(items))

nlp = load_nlp_model()
disease_list = load_disease_dict(DISEASE_DICT_PATH)
disease_set: Set[str] = set(disease_list)

# ---------------------- Helper Functions ----------------------
def clean_text(text:str)->str:
    t=str(text or "").lower()
    t=re.sub(r"[\[\]\(\)\{\}\|_,-/;:/\\]+"," ",t)
    t=re.sub(r"[^a-z0-9\-\s%+]"," ",t)
    return re.sub(r"\s+"," ",t).strip()

ABBR_MAP = {
    "bp":"blood pressure","dm":"diabetes mellitus","htn":"hypertension",
    "tb":"tuberculosis","uti":"urinary tract infection","ckd":"chronic kidney disease",
    "mi":"myocardial infarction","covid":"covid-19","hba1c":"hba1c"
}

def expand_abbrevs(text:str)->str:
    return " ".join([ABBR_MAP.get(w,w) for w in str(text or "").split()])

def conditional_fuzzy_match(token:str, choices:List[str], threshold:int):
    if not token: return (None,0)
    match = process.extractOne(token, choices, scorer=fuzz.token_sort_ratio)
    return (match[0], int(match[1])) if match else (None,0)

def remove_redundant_diseases(disease_list:List[str])->List[str]:
    diseases=sorted(set([d.lower().strip() for d in disease_list if d.strip()]), key=len, reverse=True)
    final=[]
    for d in diseases:
        if any(d in kept for kept in final): continue
        final.append(d)
    return final

def extract_conditions(text:str, mode:str="clinical", ngram_max:int=4)->List[Dict]:
    threshold = FUZZ_THRESHOLD.get(mode,88)
    results={}
    raw = clean_text(expand_abbrevs(text))
    tokens = [t for t in raw.split() if t not in STOPWORDS]
    ngrams=[" ".join(tokens[i:j]) for i in range(len(tokens)) for j in range(i+1,min(len(tokens),i+ngram_max)+1)]

    # Exact match
    for phrase in set(ngrams+tokens):
        if phrase in disease_set: results[phrase]={"disease":phrase,"source":"exact","score":100}

    # NER + Fuzzy
    try: doc=nlp(raw); ner_candidates=[ent.text.strip().lower() for ent in getattr(doc,"ents",[]) if len(ent.text)>2]
    except Exception: ner_candidates=[]
    for cand in set(ner_candidates):
        match, score = conditional_fuzzy_match(cand,disease_list,threshold)
        if match and score>=threshold: results[match]={"disease":match,"source":"ner_fuzzy","score":score}

    # Fuzzy Ngrams
    for phrase in set(ngrams):
        if phrase in results: continue
        match, score = conditional_fuzzy_match(phrase,disease_list,threshold)
        if match and score>=threshold: results[match]={"disease":match,"source":"fuzzy","score":score}

    final=[r for r in results.values() if len(r["disease"])>=3 and r["disease"] not in STOPWORDS]
    unique = remove_redundant_diseases([r["disease"] for r in final])
    return sorted([r for r in final if r["disease"] in unique], key=lambda x:(x["score"],len(x["disease"])), reverse=True)

# ---------------------- Input Section ----------------------
col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    input_mode = st.radio("Choose Input Mode:", ["📝 Manual Text Entry","📂 Upload / Local Path"], horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div class='muted'>Tips</div>", unsafe_allow_html=True)
    st.markdown("<ul class='small'><li>Use local path if uploads fail</li><li>Rename files to remove spaces</li><li>Large files may be slow</li></ul>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Manual Text Entry ----------------------
if input_mode=="📝 Manual Text Entry":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("manual_form"):
        user_text=st.text_area("Enter clinical note:", height=220, placeholder="Patient has BP, diabetes, fatty liver...")
        mode=st.selectbox("Matching Mode", ["clinical","strict","smart","lenient","exact"], index=0)
        submitted=st.form_submit_button("🔍 Extract Conditions")
    if submitted:
        if not user_text.strip(): st.warning("⚠️ Please enter some text.")
        else:
            with st.spinner("Extracting diseases..."):
                results=extract_conditions(user_text, mode)
            if results:
                st.success(f"✅ Found {len(results)} conditions!")
                chips=" ".join([f"<span class='result-chip'>{r['disease']}</span>" for r in results])
                st.markdown(chips, unsafe_allow_html=True)
                st.write(results)
            else: st.info("No diseases or conditions detected.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- File Upload / Local Path ----------------------
elif input_mode=="📂 Upload / Local Path":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Load file (Upload or Local Path)")
    file_choice=st.radio("", ["Upload File","Use Local Path"], horizontal=True)
    df=None

    if file_choice=="Upload File":
        uploaded_file=st.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])
        if uploaded_file:
            try:
                ext=uploaded_file.name.split(".")[-1].lower()
                data=uploaded_file.read()
                uploaded_file.seek(0)
                df=pd.read_csv(io.BytesIO(data)) if ext=="csv" else pd.read_excel(io.BytesIO(data), engine="openpyxl")
            except Exception as e: st.error(f"Error reading file: {e}")

    else:
        local_path=st.text_input("Enter absolute local path")
        if local_path:
            p=Path(local_path)
            if not p.exists(): st.error("File not found")
            else:
                try: df=pd.read_csv(p) if p.suffix.lower()==".csv" else pd.read_excel(p, engine="openpyxl")
                except Exception as e: st.error(f"Error reading file: {e}")

    if df is not None:
        st.success(f"✅ Loaded file — {len(df)} rows")
        st.dataframe(df.head(), use_container_width=True)
        text_cols=[c for c in df.columns if df[c].dtype==object]
        suggested=None
        hints=["note","notes","remarks","history","description","comment","text"]
        for c in text_cols:
            low=" ".join(df[c].astype(str).head(20).str.lower())
            if any(h in low for h in hints): suggested=c; break
        column=st.selectbox("Select text column:", options=text_cols, index=text_cols.index(suggested) if suggested in text_cols else 0)
        mode=st.selectbox("Matching Mode", ["clinical","strict","smart","lenient","exact"], index=0)
        if st.button("🚀 Run Extraction"):
            with st.spinner("Extracting..."):
                df["Extracted_Diseases"]=df[column].astype(str).apply(lambda x:", ".join([r["disease"] for r in extract_conditions(x, mode)]) if str(x).strip() else "")
            st.success("✅ Extraction complete")
            st.dataframe(df.head(10), use_container_width=True)
            out_name="Extracted_Results.xlsx"
            df.to_excel(out_name,index=False)
            with open(out_name,"rb") as f: st.download_button("💾 Download Excel", data=f, file_name=out_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("👨‍⚕️ **Developed by Debarghya Kundu** — polished, professional UI")
