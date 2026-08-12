import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from openai import OpenAI
from sarvamai import SarvamAI
# ============================================================
# LOAD ENV VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env")
    st.stop()

if not SARVAM_API_KEY:
    st.error("SARVAM_API_KEY not found in .env")
    st.stop()

# ============================================================
# OPENAI CLIENT (CLASSIFICATION)
# ============================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

# ============================================================
# SARVAM CLIENT (GENERATION)
# ============================================================

sarvam_client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Customer Support Assistant")

st.title("Customer Support Assistant")
st.write("BM25 Retrieval + OpenAI Tone Classification + Sarvam Generation")

# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = r"C:\Users\AadharJain\Desktop\IBM_Training-main\DAY_12\Complaint Dataset.xlsx"

try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(text):
    text = str(text)
    return re.findall(r"\b\w+\b", text.lower())

# ============================================================
# CREATE BM25 INDEX
# ============================================================

corpus = [
    tokenize_text(str(text))
    for text in df["Trouble"]
]

bm25 = BM25Okapi(corpus)

# ============================================================
# TONE CLASSIFICATION USING OPENAI
# ============================================================

ALLOWED_CATEGORIES = [
    "Strict",
    "Friendly",
    "All"
]

def classify_query_with_llm(query):

    prompt = f"""
                You are a customer complaint tone classifier.

                Classify the complaint into EXACTLY ONE category.

                Categories:

                Strict:
                - General enquiries
                - Simple issues
                - Routine complaints
                - Direct requests

                Friendly:
                - Frustration
                - Urgency
                - Repeated complaints
                - Emotional dissatisfaction
                - Escalation situations

                All:
                - Unclear
                - Multiple intents

                User Complaint:
                {query}

                Return ONLY JSON:

                {{
                    "category": "Strict"
                }}
            """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You classify customer complaint tone."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    try:
        result = json.loads(content)

        category = result.get(
            "category",
            "All"
        )

        if category not in ALLOWED_CATEGORIES:
            category = "All"

        return category

    except:
        return "All"

# ============================================================
# BM25 RETRIEVAL
# ============================================================

def retrieve_relevant_rows(query, top_k=3):

    tokenized_query = tokenize_text(query)

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []

    for idx in top_indices:

        row = df.iloc[idx]

        results.append({
            "score": float(scores[idx]),
            "trouble": str(row["Trouble"]),
            "category": str(row["Category"]),
            "solution": str(row["Solution"]),
            "alternate_solution": str(row["Alternate Solution"]),
            "company_response": str(row["Company Response"])
        })

    return results

# ============================================================
# SARVAM RESPONSE GENERATION
# ============================================================

def generate_customer_response(user_query, retrieved_rows, tone):
    context = ""

    for i, row in enumerate(retrieved_rows, start=1):
        context += f"""
                    Issue {i}

                    Trouble:
                    {row['trouble']}

                    Category:
                    {row['category']}

                    Solution:
                    {row['solution']}

                    Alternate Solution:
                    {row['alternate_solution']}

                    Company Response:
                    {row['company_response']}

                    ------------------------------------
                """

    prompt = f"""
                You are an expert customer support executive.

                Customer Complaint:
                {user_query}

                Detected Tone:
                {tone}

                Relevant Support Knowledge:
                {context}

                Instructions:

                If tone = Strict:
                - Professional
                - Direct
                - Concise

                If tone = Friendly:
                - Empathetic
                - Polite
                - Reassuring
                - Helpful

                If tone = All:
                - Balanced response

                Use:
                1. Solution
                2. Alternate Solution
                3. Company Response

                Generate a complete customer support response.
                Response should be in a paragraph.

                Do not mention the internal dataset.
            """

    response = sarvam_client.chat.completions(
        model="sarvam-105b",
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": "You are a professional customer support assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# ============================================================
# USER INPUT
# ============================================================

query = st.text_area(
    "Enter your complaint:",
    height=150
)

if st.button("Generate Response"):

    if not query.strip():
        st.warning("Please enter a complaint.")
        st.stop()

    # --------------------------------------------------------
    # CLASSIFY
    # --------------------------------------------------------

    with st.spinner("Analyzing complaint tone..."):
        detected_tone = classify_query_with_llm(query)

    st.success(
        f"LLM response Tone: {detected_tone}"
    )

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    with st.spinner("Searching complaint database..."):
        retrieved_rows = retrieve_relevant_rows(query, top_k=3)

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with st.spinner("Generating customer response..."):
        answer = generate_customer_response(
            query,
            retrieved_rows,
            detected_tone
        )

    st.subheader("Customer Response")

    st.write(answer)

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    st.subheader("Retrieved Matches")

    for i, row in enumerate(retrieved_rows, start=1):

        with st.expander(
            f"Match {i} | Score: {row['score']:.2f}"
        ):

            st.write("**Trouble**")
            st.write(row["trouble"])

            st.write("**Category**")
            st.write(row["category"])

            st.write("**Solution**")
            st.write(row["solution"])

            st.write("**Alternate Solution**")
            st.write(row["alternate_solution"])

            st.write("**Company Response**")
            st.write(row["company_response"])