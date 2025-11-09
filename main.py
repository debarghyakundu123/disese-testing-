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
# Use a wide layout and a custom title with an emoji
st.set_page_config(page_title="🧠 Smart Clinical Extractor", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TORCH_LOGS"] = "none"

# --- SESSION STATE INITIALIZATION ---
if 'batch_df' not in st.session_state:
    st.session_state['batch_df'] = None
if 'manual_results' not in st.session_state:
    st.session_state['manual_results'] = None
if 'run_status' not in st.session_state:
    # Status can be 'ready', 'complete', 'no_results', 'error'
    st.session_state['run_status'] = 'ready'
if 'uploaded_df' not in st.session_state:
    st.session_state['uploaded_df'] = None

# --- FIXED MODE DEFINITION ---
# The user requested to remove mode selection and fix it to 'clinical'
mode = "clinical"

# ---------------------- Config (MUST BE EXACT) ----------------------
DISEASE_DICT_PATH = "updatedcleaned_diseases.txt"
USE_SCISPACY = False
SCISPACY_MODEL = "en_core_sci_sm"
SPACY_MODEL = "en_core_web_sm"
FUZZ_THRESHOLD = {"exact": 100, "strict": 95, "clinical": 88, "smart": 85, "lenient": 75}
STOPWORDS = set([
    "a", "of", "the", "and", "for", "with", "in", "on", "is", "are", "by", "to", "from", 
    "patient", "history"
])

# ---------------------- Load Models (MUST BE EXACT) ----------------------
@st.cache_resource
def load_nlp_model():
    try:
        model = SCISCIPACY_MODEL if USE_SCISPACY else SPACY_MODEL
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

# ---------------------- Helper Functions (MUST BE EXACT) ----------------------
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

    # -------------------- STEP 1: Exact match --------------------
    for phrase in set(ngrams + tokens):
        if phrase in disease_set:
            results[phrase] = {"disease": phrase, "source": "exact", "score": 100}

    # -------------------- STEP 2: NER + fuzzy --------------------
    doc = nlp(raw)
    ner_candidates = [ent.text.strip().lower() for ent in getattr(doc, "ents", []) if len(ent.text) > 2]
    for cand in set(ner_candidates):
        match, score = conditional_fuzzy_match(cand, disease_list, threshold)
        if match and score >= threshold:
            results[match] = {"disease": match, "source": "ner_fuzzy", "score": score}

    # -------------------- STEP 3: Fuzzy n-grams --------------------
    for phrase in set(ngrams):
        if phrase in results:
            continue
        match, score = conditional_fuzzy_match(phrase, disease_list, threshold)
        if match and score >= threshold:
            results[match] = {"disease": match, "source": "fuzzy", "score": score}

    # -------------------- 🧠 STEP 4: SMART FALLBACK --------------------
    # If model found nothing, scan raw text for any known disease terms manually
    if not results:
        fallback_matches = [d for d in disease_list if d in raw]
        for match in fallback_matches:
            results[match] = {"disease": match, "source": "fallback", "score": 70}

    # -------------------- STEP 5: Clean and finalize --------------------
    final = [r for r in results.values() if len(r["disease"]) >= 3 and r["disease"] not in STOPWORDS]
    unique = remove_redundant_diseases([r["disease"] for r in final])
    clean_results = [r for r in final if r["disease"] in unique]
    return sorted(clean_results, key=lambda x: (x["score"], len(x["disease"])), reverse=True)

# ---------------------- Streamlit UI (Beautified & Fixed Mode) ----------------------

# --- Sidebar for Configuration and Info ---
st.sidebar.title("⚙️ Extraction Settings")
st.sidebar.markdown(
    """
    The matching mode is **fixed to Clinical** (88% threshold)
    for optimal performance on typical medical narratives.
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("Model & Mode Details")
st.sidebar.info(
    f"""
    **Model Used:** `{SCISPACY_MODEL if USE_SCISPACY else SPACY_MODEL}`
    
    **Active Mode:** **Clinical**
    
    **Threshold:** {FUZZ_THRESHOLD['clinical']}% (Fuzziness Score)
    """
)


# --- Main Content Area Header ---
st.markdown("<h1 style='text-align: center; color: #1f78b4;'>🧠 Clinical Entity Extractor</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 30px;'>
        **Intelligent extraction of medical conditions and diseases from unstructured clinical text or data files.**
    </div>
    """, 
    unsafe_allow_html=True
)

# Use columns for radio selection to center it
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    option = st.radio(
        "Choose Input Source:", 
        ["📝 Manual Text Entry", "📂 Upload File"], 
        horizontal=True,
        key="input_option"
    )

st.markdown("---") # Visual separator

# --- MAIN LEFT/RIGHT SPLIT ---
input_col, output_col = st.columns([1, 1])

# =========================================================================
# LEFT COLUMN: INPUT AND RUN
# =========================================================================
with input_col:
    # Manual Text Mode
    if option == "📝 Manual Text Entry":
        
        with st.container(border=True):
            st.subheader(f"📝 Text Input (Mode: {mode.capitalize()})")
            st.markdown("Paste your clinical note or unstructured text below.")

            with st.form("manual_form"):
                user_text = st.text_area(
                    "Clinical Text:",
                    height=250,
                    placeholder="e.g., Patient is a 55-year-old male with a history of HTN, DM, and recent MI. Also documented CKD stage 3. Requires follow-up for Hba1c.",
                    label_visibility="collapsed"
                )
                
                submitted = st.form_submit_button("🚀 Run Extraction", type="primary", use_container_width=True)

            if submitted:
                if not user_text.strip():
                    st.session_state['run_status'] = 'error'
                    st.error("⚠️ Please enter some text into the box to begin extraction.")
                else:
                    with st.spinner(f"Processing text with **{mode.capitalize()}** mode..."):
                        # --- CORE LOGIC ---
                        results = extract_conditions(user_text, mode=mode)
                        # --- END CORE LOGIC ---
                    
                    if results:
                        st.session_state['manual_results'] = results
                        st.session_state['batch_df'] = None
                        st.session_state['run_status'] = 'complete'
                    else:
                        st.session_state['run_status'] = 'no_results'
                    st.success("Extraction command sent to results panel.")
    
    # File Upload Mode
    elif option == "📂 Upload File":
        
        # Reset uploaded_df if user switches back and forth
        df = st.session_state['uploaded_df']
        
        with st.container(border=True):
            st.subheader(f"📂 Batch Extraction (Mode: {mode.capitalize()})")
            
            upload_col, path_col = st.columns(2)
            
            # --- File Upload ---
            with upload_col:
                st.markdown("##### 📤 Option A: Upload File")
                uploaded_file = st.file_uploader(
                    "Upload a CSV or Excel file", 
                    type=["csv", "xlsx"], 
                    accept_multiple_files=False,
                    label_visibility="collapsed"
                )
            
            # --- Local Path Input ---
            with path_col:
                st.markdown("##### 💻 Option B: Use Local Path")
                local_path = st.text_input(
                    "Enter full file path (Local Only)", 
                    placeholder="e.g. /data/clinical_notes.csv",
                    label_visibility="collapsed"
                )

            # --- File Loading Logic ---
            if uploaded_file and (df is None or uploaded_file.name != getattr(df, 'name', '')):
                try:
                    file_ext = uploaded_file.name.split(".")[-1].lower()
                    file_bytes = uploaded_file.read()
                    uploaded_file.seek(0)

                    if file_ext == "csv":
                        df = pd.read_csv(io.BytesIO(file_bytes))
                    elif file_ext == "xlsx":
                        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
                    else:
                        st.session_state['run_status'] = 'error'
                        st.error("❌ Unsupported file type. Please upload CSV or XLSX.")
                    
                    if df is not None:
                        df.name = uploaded_file.name
                        st.session_state['uploaded_df'] = df
                        st.success(f"✅ File **'{uploaded_file.name}'** uploaded successfully! Rows: {len(df)}")
                        st.session_state['run_status'] = 'ready'
                except Exception as e:
                    st.session_state['run_status'] = 'error'
                    st.error(f"❌ Error reading uploaded file: {e}")

            elif local_path and (df is None or local_path != getattr(df, 'path', '')):
                path = Path(local_path)
                if not path.exists():
                    st.error("❌ Local file not found. Please check the path.")
                else:
                    try:
                        if path.suffix.lower() == ".csv":
                            df = pd.read_csv(path)
                        elif path.suffix.lower() == ".xlsx":
                            df = pd.read_excel(path, engine="openpyxl")
                        else:
                            st.error("❌ Unsupported file format.")
                        
                        if df is not None:
                            df.path = local_path # Attach path for comparison
                            st.session_state['uploaded_df'] = df
                            st.success(f"✅ Loaded file from local path! Rows: {len(df)}")
                            st.session_state['run_status'] = 'ready'
                    except Exception as e:
                        st.session_state['run_status'] = 'error'
                        st.error(f"❌ Error reading file: {e}")

        # Configuration and Run Section (Appears after file is loaded)
        if st.session_state['uploaded_df'] is not None:
            st.markdown("### Configuration and Run")
            
            with st.expander("Data Preview (First 5 Rows)", expanded=False):
                st.dataframe(st.session_state['uploaded_df'].head(), use_container_width=True)

            col_select, run_col = st.columns([3, 1])

            with col_select:
                # Use the loaded df columns for selection
                column = st.selectbox(
                    "1. Select the column containing clinical text to process:", 
                    st.session_state['uploaded_df'].columns,
                    help="Choose the column that holds the clinical narratives you want to extract from.",
                    key="text_column_selector"
                )
            
            with run_col:
                st.markdown("---") # Spacer
                if st.button("🚀 Run Batch Extraction", type="primary", use_container_width=True):
                    if column not in st.session_state['uploaded_df'].columns:
                        st.error("Please select a valid column.")
                    else:
                        temp_df = st.session_state['uploaded_df'].copy()
                        with st.spinner(f"Extracting conditions from {len(temp_df)} rows using {mode.capitalize()} mode..."):
                            # --- CORE LOGIC (MUST BE EXACT) ---
                            temp_df["Extracted_Diseases"] = temp_df[column].astype(str).apply(
                                lambda x: ", ".join([r["disease"] for r in extract_conditions(x, mode)]) if x.strip() else ""
                            )
                            # --- END CORE LOGIC ---

                        st.session_state['batch_df'] = temp_df
                        st.session_state['manual_results'] = None
                        st.session_state['run_status'] = 'complete'
                        st.success("Extraction command sent to results panel.")
# =========================================================================
# RIGHT COLUMN: RESULTS
# =========================================================================
with output_col:
    st.markdown("<h3 style='text-align: center;'>📋 Extraction Results</h3>", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state['run_status'] == 'complete':
        st.success("✅ Extraction complete! See results below.")

        # --- MANUAL MODE RESULTS ---
        if st.session_state['manual_results']:
            results = st.session_state['manual_results']
            with st.expander(f"Extracted {len(results)} Conditions (Mode: {mode.capitalize()})", expanded=True):
                st.dataframe(
                    results, 
                    use_container_width=True,
                    column_config={
                        "disease": st.column_config.TextColumn("Condition"),
                        "source": st.column_config.TextColumn("Source Method"),
                        "score": st.column_config.ProgressColumn("Match Score", format="%d", min_value=0, max_value=100),
                    }
                )
        
        # --- BATCH MODE RESULTS ---
        elif st.session_state['batch_df'] is not None:
            df = st.session_state['batch_df']
            
            st.markdown(
                "<p style='text-align: center;'>The new <strong>Extracted_Diseases</strong> column has been added to your dataset. Preview of the top 10 rows:</p>",
                unsafe_allow_html=True
            )
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Download and Save Info (placed in columns for alignment)
            dl_col, save_col = st.columns([1, 1])
            
            with dl_col:
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results (CSV)",
                    data=csv_data,
                    file_name="extracted_clinical_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with save_col:
                output_path = Path.cwd() / "Extracted_Results.xlsx"
                df.to_excel(output_path, index=False)
                st.info(f"📁 **File written to server disk:** {output_path.name}") 

    elif st.session_state['run_status'] == 'no_results':
        st.info("🔎 No diseases or conditions were detected based on the current input and settings.")
    
    elif st.session_state['run_status'] == 'ready':
        st.info("Ready for input. Extracted results will appear here after running the extraction.")
    
    # Error messages are generally displayed immediately in the input column, but we keep this for robust status handling
    elif st.session_state['run_status'] == 'error':
         st.warning("Please resolve the error in the left panel to proceed.")


st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: #777;'>
        👨‍⚕️ Developed by Debarghya Kundu | AI-powered Clinical Text Extractor
    </p>
    """, 
    unsafe_allow_html=True
)
