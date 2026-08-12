import os
import re
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from openai import OpenAI
from sarvamai import SarvamAI
from guardrails_semantic import run_input_guardrails, mask_pii
import requests
from guardrails_cache import check_query_cache, save_query_cache
# ============================================================
# CONFIG
# ============================================================


st.set_page_config(
    page_title="Customer Support Assistant",
    layout="wide"
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found")
    st.stop()

if not SARVAM_API_KEY:
    st.error("SARVAM_API_KEY not found")
    st.stop()

# ============================================================
# CLIENTS
# ============================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

sarvam_client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)

# ============================================================
# SESSION STATE
# ============================================================

if "voice_query" not in st.session_state:
    st.session_state.voice_query = ""

if "detected_language" not in st.session_state:
    st.session_state.detected_language = "en"

# ============================================================
# DATA
# ============================================================

DATA_PATH = r"C:\Users\AadharJain\Desktop\IBM_Training-main\DAY_12\Complaint Dataset.xlsx"

try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ============================================================
# tokenize
# ============================================================

def tokenize_text(text):
    return re.findall(r"\b\w+\b", str(text).lower())

# ============================================================
# CLASSIFICATION
# ============================================================

def classify_query(query):

    prompt = f"""
You are a multilingual customer support classifier.

The customer may write in any language.

Tasks:

1. Identify category
2. Identify tone
3. Create an English retrieval query for BM25

Categories:
- returns
- delivery
- payment
- account
- technical
- offers
- others

Tones:
- Strict
- Friendly
- Neutral

Complaint:
{query}

Return ONLY JSON.

{{
  "retrieval_query":"payment failed transaction declined",
  "category":"payment",
  "tone":"Strict"
}}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You classify customer support complaints."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    try:

        content = response.choices[0].message.content

        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

        return json.loads(content)

    except:

        return {
            "retrieval_query": query,
            "category": "others",
            "tone": "Neutral"
        }


# ============================================================
# FILTER
# ============================================================

def filter_dataset_by_category(df, category):

    filtered_df = df[
        df["Category"]
        .astype(str)
        .str.lower()
        == category.lower()
    ]

    if len(filtered_df) == 0:
        return df

    return filtered_df


# ============================================================
# BM25
# ============================================================

def create_bm25_index(filtered_df):

    corpus = []

    for _, row in filtered_df.iterrows():

        text = f"""
        {row['Trouble']}
        {row['Solution']}
        {row['Alternate Solution']}
        """

        corpus.append(text)

    tokenized = [
        tokenize_text(doc)
        for doc in corpus
    ]

    return BM25Okapi(tokenized)


def retrieve_documents(
    query,
    bm25,
    filtered_df,
    top_k=3
):

    query_tokens = tokenize_text(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[
        -min(top_k, len(filtered_df)):
    ][::-1]

    filtered_df = filtered_df.reset_index(drop=True)

    results = []

    for idx in top_indices:

        row = filtered_df.iloc[idx]

        results.append({
            "trouble": row["Trouble"],
            "category": row["Category"],
            "solution": row["Solution"],
            "alternate_solution": row["Alternate Solution"],
            "company_response": row["Company Response"],
            "score": float(scores[idx])
        })

    return results


# ============================================================
# GENERATION
# ============================================================

def generate_customer_response(
    query,
    retrieved_docs,
    tone,
    language
):

    context = ""

    for doc in retrieved_docs:

        context += f"""
Trouble:
{doc['trouble']}

Solution:
{doc['solution']}

Alternate Solution:
{doc['alternate_solution']}

Company Response:
{doc['company_response']}

-----------------------------------
"""

    prompt = f"""
You are a customer support executive.

Customer Complaint:
{query}

Detected Tone:
{tone}

Knowledge Base:
{context}

Instructions:

Customer Language:
{language}

IMPORTANT:
Respond in exactly the same language
as the customer. Do not print customer's personal information. Tell customer you cannot
do that.

Examples:

en-IN -> English
hi-IN -> Hindi
ta-IN -> Tamil
te-IN -> Telugu
kn-IN -> Kannada
ml-IN -> Malayalam
mr-IN -> Marathi
gu-IN -> Gujarati
bn-IN -> Bengali
pa-IN -> Punjabi

If tone is Strict:
- empathetic
- apologetic
- reassuring

If tone is Friendly:
- concise
- direct
- professional

If tone is Neutral:
- balanced

Use ONLY the knowledge base.

Do not invent information.
"""
    
     
    url = "https://api.sarvam.ai/v1/chat/completions"
 
    headers = {
    "api-subscription-key": SARVAM_API_KEY,
    "Content-Type": "application/json"
    }
 
    payload = {
    "model": "sarvam-105b",
    "messages": [
        {
            "role": "system",
            "content": "You generate accurate customer support responses."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    "reasoning_effort": None,
}
    response = requests.post(
                        url,
                        headers=headers,
                        json=payload
            )
 
    if response.status_code != 200:
        st.error(f"Sarvam API Error: {response.text}")
        return "Unable to generate response."
 
    result = response.json()
 
    message = result["choices"][0]["message"]
    answer = message.get("content")
 
    return answer
 


# ============================================================
# UI
# ============================================================

st.title("Customer Support Assistant")

st.write(
    "Voice + BM25 + OpenAI Classification + Sarvam Response"
)

# ------------------------------------------------------------
# VOICE
# ------------------------------------------------------------

st.subheader("Voice Complaint")

audio_file = st.audio_input(
    "Record your complaint"
)

if audio_file is not None:

    st.audio(audio_file)

    if st.button("Transcribe Voice"):

        with st.spinner("Transcribing..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp:

                tmp.write(audio_file.getvalue())
                audio_path = tmp.name

            with open(audio_path, "rb") as f:

                response = sarvam_client.speech_to_text.transcribe(
                    file=f,
                    model="saarika:v2.5"
                )

            transcript = response.transcript
            language = response.language_code

            st.session_state.voice_query = transcript
            st.session_state.detected_language = language

            st.success("Transcription Complete")

            st.write("Transcript:")
            st.write(transcript)

            st.write(
                f"Detected Language: {language}"
            )


# ------------------------------------------------------------
# TEXT INPUT
# ------------------------------------------------------------

text_query = st.text_area(
    "Or type your complaint"
)

query = (
    st.session_state.voice_query
    if st.session_state.voice_query
    else text_query
)
 
if query:

    language = st.session_state.detected_language

    pii_found, masked_query = mask_pii(query)

    if pii_found:
        st.subheader("Masked Query")
        st.write(masked_query)

    valid, message = run_input_guardrails(masked_query)

    if not valid:
        st.error(message)
        st.stop()

    cache_hit, cache_data = check_query_cache(masked_query)

    if cache_hit:

        st.subheader("Cached Answer")
        st.write(cache_data)
        st.stop()

    question_embedding = cache_data

    with st.spinner("Thinking..."):

        classification = classify_query(query)

        retrieval_query = classification["retrieval_query"]
        category = classification["category"]
        tone = classification["tone"]

        filtered_df = filter_dataset_by_category(
            df,
            category
        )

        bm25 = create_bm25_index(filtered_df)

        retrieved_docs = retrieve_documents(
            retrieval_query,
            bm25,
            filtered_df,
            top_k=3
        )

        if not retrieved_docs:
            st.warning("No relevant documents found.")
            st.stop()

        final_answer = generate_customer_response(
            query,
            retrieved_docs,
            tone,
            language
        )

    save_query_cache(
        masked_query,
        question_embedding,
        final_answer
    )

    st.subheader("Answer")
    st.write(final_answer)

    for i, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        with st.expander(f"Result {i}"):

            st.write(
                f"Score: {doc['score']:.2f}"
            )

            st.write(
                f"Category: {doc['category']}"
            )

            st.write(
                f"Trouble: {doc['trouble']}"
            )

            st.write(
                f"Solution: {doc['solution']}"
            )

            st.write(
                f"Alternate Solution: {doc['alternate_solution']}"
            )

            st.write(
                f"Company Response: {doc['company_response']}"
            )