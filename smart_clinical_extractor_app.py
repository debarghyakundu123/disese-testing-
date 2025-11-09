# smart_clinical_extractor_app.py
# 🚀 Streamlit Clinical Extractor — Optimized for Cloud Deployment

import streamlit as st
import pandas as pd
import re, io, os, warnings
from pathlib import Path
from typing import List, Dict, Set
import spacy
from rapidfuzz import process, fuzz

# ==============================================================
# 🌐 Streamlit App Configuration
# ==============================================================
st.set_page_config(
    page_title="🧠 Smart Clinical Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize session state variables
for key, val in {
    "batch_df": None,
    "manual_results": None,
    "uploaded_df": None,
    "run_status": "ready",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==============================================================
# ⚙️ Configuration Parameters
# ==============================================================
DISEASE_DICT_PATH = "updatedcleaned_diseases.txt"
SPACY_MODEL = "en_core_web_sm"
FUZZ_THRESHOLD = {"clinical": 88}
STOPWORDS = set(["a", "of", "the", "and", "for", "with", "in", "on", "is", "are", "by", "to", "from", "patient", "history"])
ABBR_MAP = {
    "bp": "blood pressure", "dm": "diabetes mellitus", "htn": "hypertension",
    "tb": "tuberculosis", "uti": "urinary tract infection", "ckd": "chronic kidney disease",
    "mi": "myocardial infarction", "covid": "covid-19", "hba1c": "hba1c"
}

# ==============================================================
# 🧠 Lazy Loading (Optimized for Streamlit Cloud)
# ==============================================================
@st.cache_resource(show_spinner=False)
def load_nlp():
    """Load spaCy model safely (smallest one)."""
    try:
        return spacy.load(SPACY_MODEL, disable=["parser", "tagger", "lemmatizer"])
    except Exception as e:
        st.warning(f"⚠️ NLP model not found, downloading... ({e})")
        from spacy.cli import download
        download(SPACY_MODEL)
        return spacy.load(SPACY_MODEL, disable=["parser", "tagger", "lemmatizer"])

@st.cache_data(show_spinner=False)
def load_disease_dict(path: str):
    """Load and clean disease dictionary."""
    p = Path(path)
    if not p.exists():
        st.error(f"❌ Disease dictionary not found at {path}")
        st.stop()
    with p.open("r", encoding="utf-8") as f:
        items = [line.strip().lower() for line in f if line.strip()]
    return sorted(set(items))

nlp = load_nlp()
disease_list = load_disease_dict(DISEASE_DICT_PATH)
disease_set = set(disease_list)

# ==============================================================
# 🧩 Utility Functions
# ==============================================================
def clean_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s%+]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def expand_abbrevs(text: str) -> str:
    return " ".join([ABBR_MAP.get(w, w) for w in text.split()])

def fuzzy_match(token: str, choices: List[str], threshold: int = 88):
    if not token:
        return None, 0
    match = process.extractOne(token, choices, scorer=fuzz.token_sort_ratio)
    return (match[0], match[1]) if match and match[1] >= threshold else (None, 0)

def extract_conditions(text: str, mode="clinical"):
    threshold = FUZZ_THRESHOLD[mode]
    text = clean_text(expand_abbrevs(text))
    tokens = [t for t in text.split() if t not in STOPWORDS]

    results = {}
    doc = nlp(text)

    # Exact match
    for token in tokens:
        if token in disease_set:
            results[token] = {"disease": token, "source": "exact", "score": 100}

    # NER-based fuzzy
    for ent in doc.ents:
        match, score = fuzzy_match(ent.text.lower(), disease_list, threshold)
        if match:
            results[match] = {"disease": match, "source": "ner_fuzzy", "score": score}

    # Smart fallback
    if not results:
        for d in disease_list:
            if d in text:
                results[d] = {"disease": d, "source": "fallback", "score": 70}

    return sorted(results.values(), key=lambda x: x["score"], reverse=True)

# ==============================================================
# 🎨 Streamlit UI
# ==============================================================
st.sidebar.title("⚙️ Extraction Settings")
st.sidebar.markdown("**Mode:** Clinical (88%) — optimized for unstructured medical text")
st.sidebar.info(f"Model: `{SPACY_MODEL}`")

st.markdown("<h1 style='text-align:center;'>🧠 Smart Clinical Extractor</h1>", unsafe_allow_html=True)
st.write("Extract diseases and medical conditions from clinical notes using fuzzy NLP matching.")

mode = "clinical"
input_type = st.radio("Choose Input Type:", ["📝 Manual Text", "📂 Upload File"], horizontal=True)
st.markdown("---")

# ==============================================================
# ✏️ Manual Text Mode
# ==============================================================
if input_type == "📝 Manual Text":
    text = st.text_area("Enter Clinical Text:", height=200, placeholder="e.g., Patient with HTN, DM, and recent MI...")
    if st.button("🚀 Extract Conditions"):
        if text.strip():
            with st.spinner("Extracting conditions..."):
                results = extract_conditions(text)
            if results:
                st.success(f"✅ {len(results)} conditions found.")
                st.dataframe(results)
            else:
                st.info("No diseases or conditions detected.")
        else:
            st.warning("Please enter some text.")

# ==============================================================
# 📁 File Upload Mode
# ==============================================================
else:
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        column = st.selectbox("Select column with clinical text:", df.columns)
        if st.button("🚀 Run Batch Extraction"):
            with st.spinner("Processing rows..."):
                df["Extracted_Diseases"] = df[column].astype(str).apply(
                    lambda x: ", ".join([r["disease"] for r in extract_conditions(x)]) if x.strip() else ""
                )
            st.success("✅ Extraction complete!")
            st.dataframe(df.head(10))
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results (CSV)", csv, "extracted_results.csv", "text/csv")

# ==============================================================
# 🩺 Footer
# ==============================================================
st.markdown("---")
st.caption("👨‍⚕️ Developed by **Debarghya Kundu** | AI-powered Clinical Text Extractor")

