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

import streamlit as st
import pandas as pd
import re
from rapidfuzz import process, fuzz

# =============== NLP MODEL LOADING ==================
@st.cache_resource
def load_nlp():
    import spacy

    model = SCISPACY_MODEL if USE_SCISPACY else SPACY_MODEL
    try:
        nlp = spacy.load(model, disable=["parser", "tagger"])
        return nlp
    except Exception as e:
        st.error(f"❌ Could not load spaCy model: {e}")
        st.stop()

nlp = load_nlp()

# =============== LOAD DISEASE DICTIONARY ==================
@st.cache_resource
def load_dictionary():
    try:
        with open(DISEASE_DICT_PATH, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        return words
    except Exception as e:
        st.error(f"❌ Could not load dictionary file: {e}")
        st.stop()

DISEASE_LIST = load_dictionary()

# =============== CLEAN TEXT ==================
def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower().strip()

# =============== FUZZY MATCHING ==================
def fuzzy_extract(text, mode="smart"):
    threshold = FUZZ_THRESHOLD.get(mode, 85)
    extracted = []

    for word in DISEASE_LIST:
        score = fuzz.partial_ratio(text.lower(), word.lower())
        if score >= threshold:
            extracted.append((word, score))

    return sorted(list(set([w for w, s in extracted])))

# =============== NLP NER MATCHING ==================
def nlp_extract(text):
    doc = nlp(text)
    matches = []

    for ent in doc.ents:
        label = ent.label_
        if label.lower() in ["disease", "condition", "problem", "symptom"]:
            matches.append(ent.text)

    return list(set(matches))

# =============== MAIN EXTRACTOR ==================
def extract_all(text):
    cleaned = clean_text(text)
    
    fuzzy_terms = fuzzy_extract(cleaned, mode="smart")
    ner_terms = nlp_extract(text)

    combined = list(set(fuzzy_terms + ner_terms))
    return {
        "fuzzy_matches": fuzzy_terms,
        "ner_matches": ner_terms,
        "final_output": combined
    }

# =============== STREAMLIT UI ==================
st.title("🧠 Smart Clinical Extractor (spaCy + Fuzzy Matching)")

user_input = st.text_area("Enter clinical text:")

if st.button("Extract"):
    if not user_input.strip():
        st.warning("Enter some text first.")
    else:
        output = extract_all(user_input)
        st.subheader("Fuzzy Matches")
        st.write(output["fuzzy_matches"])

        st.subheader("NER Matches")
        st.write(output["ner_matches"])

        st.subheader("Final Combined Output")
        st.write(output["final_output"])
