# smart_clinical_extractor_super_ui.py
# ==================================================
# 🧠 Smart Clinical Extractor — Super UI Version
# ==================================================

import streamlit as st
import pandas as pd
import re, io, os, warnings
from pathlib import Path
from typing import List, Dict, Set
import spacy
from rapidfuzz import process, fuzz

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="🧠 Smart Clinical Extractor", layout="wide")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TORCH_LOGS"] = "none"

# ---------------------- UI Style Enhancements ----------------------
st.markdown("""
<style>
/* Background gradient */
body { background: linear-gradient(to bottom right, #0f1117, #1a1c24); color: #e0e0e0; font-family: 'Inter', sans-serif; }

/* Hero */
.hero {
    text-align:center; padding:60px 20px;
}
.hero h1 { font-size:3rem; font-weight:800; color:#4f46e5; text-shadow: 0 0 15px #4f46e5; animation: glowText 2s infinite alternate; }
.hero p { font-size:1.1rem; color:#a0a0a0; max-width:700px; margin:0 auto; line-height:1.5; }

/* Cards */
.card { background-color:#1c1c1c; border-radius:12px; padding:25px 30px; margin:20px auto; border:1px solid #2a2a2a; box-shadow:0 4px 12px rgba(0,0,0,0.5); transition: all 0.3s ease-in-out; }
.card:hover { border-color:#4f46e5; box-shadow:0 6px 18px rgba(79,70,229,0.4); transform: translateY(-2px); }

/* Buttons */
.stButton>button { background-color:#4f46e5; color:white; border:none; padding:12px 25px; font-weight:600; border-radius:6px; cursor:pointer; transition: all 0.2s ease; }
.stButton>button:hover { background-color:#4338ca; transform: translateY(-2px) scale(1.05); box-shadow:0 0 12px #4f46e5; }

/* Result chips */
.result-chip { background-color:#4f46e5; color:#fff; padding:6px 14px; margin:4px; border-radius:8px; font-weight:600; display:inline-block; animation: glowPulse 2s infinite alternate; }
.result-chip:hover { background-color:#4338ca; transform: translateY(-1px); }

/* Animations */
@keyframes glowText { 0% { text-shadow:0 0 10px #4f46e5;} 50% {text-shadow:0 0 20px #4338ca;} 100% {text-shadow:0 0 10px #4f46e5;} }
@keyframes glowPulse { 0% { box-shadow:0 0 5px #4f46e5;} 50% { box-shadow:0 0 12px #4338ca;} 100% {box-shadow:0 0 5px #4f46e5;} }

/* Breadcrumbs */
.breadcrumb { color:#9ca3af; font-size:0.9rem; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------- Config ----------------------
DISEASE_DICT_PATH = "diseases_cleaned_final.txt"
USE_SCISPACY = False
SCISPACY_MODEL = "en_core_sci_sm"
SPACY_MODEL = "en_core_web_sm"
FUZZ_THRESHOLD = {"exact":100,"strict":95,"clinical":88,"smart":85,"lenient":75}
STOPWORDS = set(["a","of","the","and","for","with","in","on","is","are","by","to","from","patient","history"])
ABBR_MAP = {"bp":"blood pressure","dm":"diabetes mellitus","htn":"hypertension",
            "tb":"tuberculosis","uti":"urinary tract infection","ckd":"chronic kidney disease",
            "mi":"myocardial infarction","covid":"covid-19","hba1c":"hba1c"}

# ---------------------- Load Models ----------------------
@st.cache_resource
def load_nlp_model():
    model = SCISPACY_MODEL if USE_SCISPACY else SPACY_MODEL
    try: return spacy.load(model, disable=["parser","tagger"])
    except Exception as e: st.error(f"❌ Failed to load NLP model: {e}"); st.stop()

@st.cache_data
def load_disease_dict(path:str) -> List[str]:
    p = Path(path)
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

    for phrase in set(ngrams+tokens):
        if phrase in disease_set: results[phrase]={"disease":phrase,"source":"exact","score":100}

    try: doc=nlp(raw); ner_candidates=[ent.text.strip().lower() for ent in getattr(doc,"ents",[]) if len(ent.text)>2]
    except Exception: ner_candidates=[]
    for cand in set(ner_candidates):
        match, score = conditional_fuzzy_match(cand,disease_list,threshold)
        if match and score>=threshold: results[match]={"disease":match,"source":"ner_fuzzy","score":score}

    for phrase in set(ngrams):
        if phrase in results: continue
        match, score = conditional_fuzzy_match(phrase,disease_list,threshold)
        if match and score>=threshold: results[match]={"disease":match,"source":"fuzzy","score":score}

    final=[r for r in results.values() if len(r["disease"])>=3 and r["disease"] not in STOPWORDS]
    unique = remove_redundant_diseases([r["disease"] for r in final])
    return sorted([r for r in final if r["disease"] in unique], key=lambda x:(x["score"],len(x["disease"])), reverse=True)

# ---------------------- Dark/Light Mode Toggle ----------------------
mode = st.sidebar.selectbox("🎨 Theme Mode", ["Dark","Light"])
if mode=="Light":
    st.markdown("<style>body{background:#f0f2f6;color:#111}</style>", unsafe_allow_html=True)

# ---------------------- Top Navigation Tabs ----------------------
tabs = st.tabs(["🏠 Home","📂 File Upload","📝 Write / Paste Text"])

# ---------------------- Home Tab ----------------------
with tabs[0]:
    st.markdown('<div class="hero"><h1>🧠 Smart Clinical Extractor</h1><p>Paste your clinical notes or upload a file — the extractor detects diseases, conditions, and key terms automatically.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home</div>', unsafe_allow_html=True)

# ---------------------- File Upload Tab ----------------------
# ---------------------- File Upload Tab ----------------------
with tabs[1]:
    st.markdown('<div class="breadcrumb">Home > File Upload</div>', unsafe_allow_html=True)
    st.header("📂 Upload File (Browser Only)")

    # Card-style container
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # ---------------- File Upload Option ----------------
    uploaded_file = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv","xlsx"],
        help="Drag & drop your file here (max 200MB). Browser upload only.",
        key="upload_file"
    )


    df = None
    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(uploaded_file.read()))
        else:
            df = pd.read_excel(io.BytesIO(uploaded_file.read()), engine="openpyxl")
        st.success(f"✅ Loaded {len(df)} rows from uploaded file")

    # ---------------- If df is loaded ----------------
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
        
        # Suggest text column
        text_cols = [c for c in df.columns if df[c].dtype == object]
        suggested = None
        hints = ["note","notes","remarks","history","description","comment","text"]
        for c in text_cols:
            low = " ".join(df[c].astype(str).head(20).str.lower())
            suggested = c if any(h in low for h in hints) else suggested

        column = st.selectbox(
            "Select text column:", 
            options=text_cols, 
            index=text_cols.index(suggested) if suggested else 0
        )

        # Matching mode
        mode_select = st.selectbox(
            "Matching Mode", 
            ["clinical","strict","smart","lenient","exact"], 
            index=0
        )

        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            st.write("Adjust fuzzy thresholds, NLP model, and stopwords (defaults used currently).")

        # Run extraction
        if st.button("🚀 Run Extraction"):
            with st.spinner("Extracting..."):
                df["Extracted_Diseases"] = df[column].astype(str).apply(
                    lambda x: ", ".join([r["disease"] for r in extract_conditions(x, mode_select)]) if str(x).strip() else ""
                )
            st.success("✅ Extraction complete")
            st.dataframe(df.head(10), use_container_width=True)

            # Download button
            out_name = "Extracted_Results.xlsx"
            df.to_excel(out_name, index=False)
            with open(out_name, "rb") as f:
                st.download_button(
                    "💾 Download Excel", 
                    data=f, 
                    file_name=out_name, 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    st.markdown('</div>', unsafe_allow_html=True)  # Close card

# ---------------------- Manual Text Entry Tab ----------------------
with tabs[2]:
    st.markdown('<div class="breadcrumb">Home > Write / Paste Text</div>', unsafe_allow_html=True)
    st.header("📝 Manual Text Entry")
    with st.form("manual_form"):
        user_text=st.text_area("Enter clinical note:", height=220, placeholder="Patient has BP, diabetes, fatty liver...")
        mode_select=st.selectbox("Matching Mode", ["clinical","strict","smart","lenient","exact"], index=0)
        submitted=st.form_submit_button("🔍 Extract Conditions")
    if submitted:
        if not user_text.strip(): st.warning("⚠️ Please enter some text.")
        else:
            with st.spinner("Extracting..."):
                results=extract_conditions(user_text, mode_select)
# After results = extract_conditions(user_text, mode_select)

            if results:
                st.success(f"✅ Found {len(results)} conditions!")


                # 2️⃣ Display a clean table
                df_results = pd.DataFrame(results)
                st.subheader("Detailed Results")
                st.dataframe(df_results, use_container_width=True)
            else:
                st.info("No diseases detected.")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("👨‍⚕️ **Developed by Debarghya Kundu** — polished, professional UI")
