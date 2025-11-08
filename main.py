# smart_clinical_extractor_app.py
# Streamlit wrapper for your existing extractor (dual mode: text + file upload)

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

# ---------------------- Config ----------------------
DISEASE_DICT_PATH = "diseases_cleaned_final.txt"
USE_SCISPACY = False
SCISPACY_MODEL = "en_core_sci_sm"
SPACY_MODEL = "en_core_web_sm"
FUZZ_THRESHOLD = {"exact": 100, "strict": 95, "clinical": 88, "smart": 85, "lenient": 75}
STOPWORDS = set([
    "a", "of", "the", "and", "for", "with", "in", "on", "is", "are", "by", "to", "from", 
    "patient", "history"
])

# ---------------------- Load Models ----------------------
@st.cache_resource
def load_nlp_model():
    try:
        model = SCISPACY_MODEL if USE_SCISPACY else SPACY_MODEL
        return spacy.load(model, disable=["parser", "tagger"])
    except Exception as e:
        st.error(f"❌ Failed to load NLP model: {e}")
        st.stop()

@st.cache_data
def load_disease_dict(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        st.error(f"❌ Disease dictionary not found at {path}")
        st.stop()
    with p.open("r", encoding="utf-8") as f:
        items = [line.strip().lower() for line in f if line.strip()]
    return sorted(set(items))

nlp = load_nlp_model()
disease_list = load_disease_dict(DISEASE_DICT_PATH)
disease_set: Set[str] = set(disease_list)

# ---------------------- Helper Functions ----------------------
def clean_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[\[\]\(\)\{\}\|_,-/;:/\\]+", " ", t)
    t = re.sub(r"[^a-z0-9\-\s%+]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

ABBR_MAP = {
    "bp": "blood pressure",
    "dm": "diabetes mellitus",
    "htn": "hypertension",
    "tb": "tuberculosis",
    "uti": "urinary tract infection",
    "ckd": "chronic kidney disease",
    "mi": "myocardial infarction",
    "covid": "covid-19",
    "hba1c": "hba1c"
}

def expand_abbrevs(text: str) -> str:
    tokens = text.split()
    return " ".join([ABBR_MAP.get(w, w) for w in tokens])

def conditional_fuzzy_match(token: str, choices: List[str], threshold: int):
    if not token:
        return (None, 0)
    match = process.extractOne(token, choices, scorer=fuzz.token_sort_ratio)
    return (match[0], int(match[1])) if match else (None, 0)

def remove_redundant_diseases(disease_list: List[str]) -> List[str]:
    diseases = sorted(set([d.lower().strip() for d in disease_list if d.strip()]), key=len, reverse=True)
    final = []
    for d in diseases:
        if any(d in kept for kept in final):
            continue
        final.append(d)
    return final

def extract_conditions(text: str, mode: str = "clinical", ngram_max: int = 4) -> List[Dict]:
    threshold = FUZZ_THRESHOLD[mode]
    results = {}
    raw = clean_text(expand_abbrevs(text))
    tokens = [t for t in raw.split() if t not in STOPWORDS]

    ngrams = []
    for i in range(len(tokens)):
        for j in range(i + 1, min(len(tokens), i + ngram_max) + 1):
            ngrams.append(" ".join(tokens[i:j]))

    # exact match
    for phrase in set(ngrams + tokens):
        if phrase in disease_set:
            results[phrase] = {"disease": phrase, "source": "exact", "score": 100}

    # NER + fuzzy
    doc = nlp(raw)
    ner_candidates = [ent.text.strip().lower() for ent in getattr(doc, "ents", []) if len(ent.text) > 2]
    for cand in set(ner_candidates):
        match, score = conditional_fuzzy_match(cand, disease_list, threshold)
        if match and score >= threshold:
            results[match] = {"disease": match, "source": "ner_fuzzy", "score": score}

    # fuzzy ngrams
    for phrase in set(ngrams):
        if phrase in results:
            continue
        match, score = conditional_fuzzy_match(phrase, disease_list, threshold)
        if match and score >= threshold:
            results[match] = {"disease": match, "source": "fuzzy", "score": score}

    final = [r for r in results.values() if len(r["disease"]) >= 3 and r["disease"] not in STOPWORDS]
    unique = remove_redundant_diseases([r["disease"] for r in final])
    clean_results = [r for r in final if r["disease"] in unique]
    return sorted(clean_results, key=lambda x: (x["score"], len(x["disease"])), reverse=True)

# ---------------------- Streamlit UI ----------------------
st.title("🧠 Smart Clinical Extractor")
st.caption("Extract medical conditions from text or uploaded files easily.")

option = st.radio("Choose Input Mode:", ["📝 Manual Text Entry", "📂 Upload File"], horizontal=True)

# Manual Text Mode
if option == "📝 Manual Text Entry":
    with st.form("manual_form"):
        user_text = st.text_area(
            "Enter clinical note or text:",
            height=200,
            placeholder="e.g., Patient has BP, diabetes, fatty liver since 2021..."
        )
        mode = st.selectbox("Matching Mode", ["clinical", "strict", "smart", "lenient", "exact"], index=0)
        submitted = st.form_submit_button("🔍 Extract Conditions")

    if submitted:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            with st.spinner("Extracting diseases..."):
                results = extract_conditions(user_text, mode=mode)
            if results:
                st.success(f"✅ Found {len(results)} conditions!")
                st.dataframe(results, use_container_width=True)
            else:
                st.info("No diseases or conditions detected.")

# File Upload Mode
# File Upload Mode
elif option == "📂 Upload File":
    st.write("### Choose how to load your file:")
    file_option = st.radio("Select file input method:", ["📤 Upload File", "💻 Use Local Path"], horizontal=True)

    df = None

    if file_option == "📤 Upload File":
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=False)
        if uploaded_file:
            try:
                file_ext = uploaded_file.name.split(".")[-1].lower()
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)

                if file_ext == "csv":
                    df = pd.read_csv(io.BytesIO(file_bytes))
                elif file_ext == "xlsx":
                    df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
                else:
                    st.error("❌ Unsupported file type. Please upload CSV or XLSX.")
                    st.stop()

                st.success(f"✅ File uploaded successfully! Rows: {len(df)}")

            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {e}")

    elif file_option == "💻 Use Local Path":
        local_path = st.text_input("Enter full file path:", placeholder="e.g. C:\\Users\\Debarghya Kundu\\Desktop\\Final_Raw.xlsx")
        if local_path:
            path = Path(local_path)
            if not path.exists():
                st.error("❌ File not found. Please check the path.")
            else:
                try:
                    if path.suffix.lower() == ".csv":
                        df = pd.read_csv(path)
                    elif path.suffix.lower() == ".xlsx":
                        df = pd.read_excel(path, engine="openpyxl")
                    else:
                        st.error("❌ Unsupported file format.")
                        st.stop()
                    st.success(f"✅ Loaded file from local path! Rows: {len(df)}")
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")

    if df is not None:
        st.write("📊 Preview of Data:")
        st.dataframe(df.head(), use_container_width=True)

        column = st.selectbox("Select the column containing clinical text:", df.columns)
        mode = st.selectbox("Matching Mode", ["clinical", "strict", "smart", "lenient", "exact"], index=0)

        if st.button("🚀 Run Extraction"):
            with st.spinner("Extracting conditions from file..."):
                df["Extracted_Diseases"] = df[column].astype(str).apply(
                    lambda x: ", ".join([r["disease"] for r in extract_conditions(x, mode)]) if x.strip() else ""
                )
            st.success("✅ Extraction complete!")

            st.subheader("📋 Preview of Extracted Results")
            st.dataframe(df.head(), use_container_width=True)

            output_path = Path.cwd() / "Extracted_Results.xlsx"
            df.to_excel(output_path, index=False)
            st.info(f"📁 Results saved locally at: {output_path}")

            st.download_button(
                label="💾 Download Results (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="extracted_results.csv",
                mime="text/csv"
            )

st.markdown("---")
st.markdown("👨‍⚕️ **Developed by Debarghya Kundu** | AI-powered Clinical Text Extractor")
