import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from audiorecorder import audiorecorder
import tempfile

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from openai import OpenAI
from sarvamai import SarvamAI

# LOAD ENV VARIABLES

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env")
    st.stop()

if not SARVAM_API_KEY:
    st.error("SARVAM_API_KEY not found in .env")
    st.stop()


# OPENAI CLIENT (CLASSIFICATION)


openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# SARVAM CLIENT (GENERATION)


sarvam_client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)
print(dir(sarvam_client.speech_to_text))

# STREAMLIT UI


st.set_page_config(page_title="Customer Support Assistant")

st.title("Customer Support Assistant")
st.write("BM25 Retrieval + OpenAI Tone Classification + Sarvam Generation")


# LOAD DATASET


DATA_PATH = r"C:\Users\AadharJain\Desktop\IBM_Training-main\DAY_12\Complaint Dataset.xlsx"

try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()


# TOKENIZATION


def tokenize_text(text):
    return re.findall(r"\b\w+\b", str(text).lower())


# CATEGORY + TONE CLASSIFICATION


def classify_query(query):

    prompt = f"""
You are a multilingual customer support classifier.

The user may write in any language.

Tasks:

1. Determine issue category
2. Determine customer tone
3. Generate an English retrieval query suitable for BM25

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

User Complaint:
{query}

Return ONLY JSON:

{{
  "retrieval_query":
      "internet connectivity issue network outage",

  "category":
      "technical",

  "tone":
      "Friendly"
}}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You classify customer support queries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    try:
        return json.loads(
            response.choices[0].message.content
        )

    except:
        return {
            "retrieval_query": query,
            "category": "others",
            "tone": "Neutral"
        }


# FILTER DATASET BY CATEGORY


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


# CREATE BM25 INDEX


def create_bm25_index(filtered_df):

    search_texts = []

    for _, row in filtered_df.iterrows():

        search_text = f"""
        {row['Trouble']}
        {row['Solution']}
        {row['Alternate Solution']}
        """

        search_texts.append(search_text)

    tokenized_corpus = [
        tokenize_text(text)
        for text in search_texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


# BM25 RETRIEVAL


def retrieve_documents(
    query,
    bm25,
    filtered_df,
    top_k=3
):

    tokenized_query = tokenize_text(query)

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[
        -min(top_k, len(filtered_df)):
    ][::-1]

    results = []

    filtered_df = filtered_df.reset_index(drop=True)

    for idx in top_indices:

        row = filtered_df.iloc[idx]

        results.append({

            "trouble":
                row["Trouble"],

            "category":
                row["Category"],

            "solution":
                row["Solution"],

            "alternate_solution":
                row["Alternate Solution"],

            "company_response":
                row["Company Response"],

            "score":
                float(scores[idx])

        })

    return results


# SARVAM RESPONSE GENERATION


def generate_customer_response(
    query,
    retrieved_docs,
    tone
):

    context = ""

    for doc in retrieved_docs:

        context += f"""
Trouble:
{doc['trouble']}

Category:
{doc['category']}

Solution:
{doc['solution']}

Alternate Solution:
{doc['alternate_solution']}

Company Response:
{doc['company_response']}

----------------------------------------
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

Use the provided knowledge only.

Generate the final customer response.
"""

    response = sarvam_client.chat.completions(
        model="sarvam-105b",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "You are a customer support assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# STREAMLIT UI

st.subheader("Voice Complaint")

audio_bytes = audiorecorder(
    text="Click to record",
    recording_color="#e74c3c",
    neutral_color="#6aa36f",
    icon_name="microphone",
    icon_size="2x",
)

voice_query = None
detected_language = None

def transcribe_audio(audio_bytes):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(audio_bytes)
        audio_path = tmp.name

    with open(audio_path, "rb") as audio_file:

        response = sarvam_client.speech_to_text.transcribe(
            file=audio_file,
            model="saarika:v2"
        )

    transcript = response.transcript

    language = response.language_code

    return transcript, language

if audio_bytes:

    with st.spinner("Transcribing audio..."):

        transcript, detected_language = transcribe_audio(
            audio_bytes
        )

    st.success("Voice detected")

    st.write("Transcript:")
    st.write(transcript)

    st.write(
        f"Detected Language: {detected_language}"
    )

    voice_query = transcript

text_query = st.text_area(
    "Enter your complaint:"
)

query = voice_query if voice_query else text_query

if st.button("Generate Response"):

    if not query.strip():

        st.warning(
            "Please enter a complaint."
        )

    else:

        
        # CATEGORY + TONE
        

        with st.spinner(
            "Analyzing complaint..."
        ):

            classification = classify_query(query)

        category = classification["category"]
        tone = classification["tone"]

        st.info(
            f"Category: {category}"
        )

        st.info(
            f"Tone: {tone}"
        )

        
        # FILTER
        

        filtered_df = filter_dataset_by_category(
            df,
            category
        )

        st.write(
            f"Filtered records: {len(filtered_df)}"
        )

        
        # BM25
        

        bm25 = create_bm25_index(
            filtered_df
        )

        retrieved_docs = retrieve_documents(
            query,
            bm25,
            filtered_df,
            top_k=3
        )

        
        # SARVAM
        

        with st.spinner(
            "Generating response..."
        ):

            answer = generate_customer_response(
                query,
                retrieved_docs,
                tone
            )

        st.subheader(
            "Customer Response"
        )

        st.write(answer)

        
        # SOURCES
        

        st.subheader(
            "Retrieved Documents"
        )

        for i, doc in enumerate(
            retrieved_docs,
            start=1
        ):

            with st.expander(
                f"Result {i}"
            ):

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