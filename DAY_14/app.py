import os
import re
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import requests

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from openai import OpenAI

from sarvamai import SarvamAI

from guardrails_semantic import (
    run_input_guardrails,
    mask_pii
)

from guardrails_cache import (
    check_query_cache,
    save_query_cache
)


# ==========================
# RAGAS IMPORTS
# ==========================

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    AnswerCorrectness
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.testset import TestsetGenerator


from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)


# ============================================================
# ENV
# ============================================================

load_dotenv()


OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

SARVAM_API_KEY = os.getenv(
    "SARVAM_API_KEY"
)


if not OPENAI_API_KEY:

    st.error(
        "OPENAI_API_KEY missing"
    )

    st.stop()



if not SARVAM_API_KEY:

    st.error(
        "SARVAM_API_KEY missing"
    )

    st.stop()



# ============================================================
# CONFIG
# ============================================================


st.set_page_config(
    page_title="Customer Support Assistant",
    layout="wide"
)



# ============================================================
# OPENAI CLIENT
# ============================================================


openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)



# ============================================================
# SARVAM CLIENT
# ============================================================


sarvam_client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)




# ============================================================
# LANGCHAIN MODELS
# ============================================================


llm = ChatOpenAI(

    model="gpt-4o-mini",

    temperature=0

)



embeddings = OpenAIEmbeddings(

    model="text-embedding-3-small"

)



# ============================================================
# RAGAS WRAPPERS
# ============================================================


ragas_llm = LangchainLLMWrapper(
    llm
)


ragas_embeddings = LangchainEmbeddingsWrapper(
    embeddings
)



# ============================================================
# RAGAS METRICS
# ============================================================


metric_faithfulness = Faithfulness(
    llm=ragas_llm
)



metric_answer_relevancy = AnswerRelevancy(

    llm=ragas_llm,

    embeddings=ragas_embeddings

)



metric_context_precision = (
    LLMContextPrecisionWithReference(
        llm=ragas_llm
    )
)



metric_context_recall = LLMContextRecall(

    llm=ragas_llm

)



metric_answer_correctness = AnswerCorrectness(

    llm=ragas_llm,

    embeddings=ragas_embeddings

)





# ============================================================
# SESSION STATE
# ============================================================


if "voice_query" not in st.session_state:

    st.session_state.voice_query = ""



if "detected_language" not in st.session_state:

    st.session_state.detected_language = "en-IN"




# ============================================================
# DATA
# ============================================================


DATA_PATH = (
r"C:\Users\AadharJain\Desktop\IBM_Training-main\DAY_12\Complaint Dataset.xlsx"
)



try:

    df = pd.read_excel(
        DATA_PATH
    )


except Exception as e:

    st.error(
        f"Dataset loading failed: {e}"
    )

    st.stop()





# ============================================================
# TOKENIZER
# ============================================================


def tokenize_text(text):

    return re.findall(
        r"\b\w+\b",
        str(text).lower()
    )





# ============================================================
# RAGAS SYNTHETIC DATA GENERATION
# ============================================================


from langchain_core.documents import Document


@st.cache_resource
def create_ragas_testset(
    dataframe,
    test_size=20
):

    documents = []

    combined_text = ""


    for _, row in dataframe.head(test_size).iterrows():

        combined_text += f"""

Customer Issue:

{row['Trouble']}


Solution:

{row['Solution']}


Alternative Solution:

{row['Alternate Solution']}


Company Response:

{row['Company Response']}


====================================


"""


    documents.append(
        Document(
            page_content=combined_text
        )
    )


    generator = TestsetGenerator.from_langchain(

        llm,

        embeddings

    )


    testset = generator.generate_with_langchain_docs(

        documents,

        testset_size=test_size

    )


    return testset



# Load synthetic dataset once

ragas_testset = create_ragas_testset(
    df,
    test_size=20
)




# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

# ============================================================
# CLASSIFICATION
# ============================================================


def classify_query(query):


    prompt = f"""

You are a multilingual customer support classifier.


Tasks:

1. Identify complaint category
2. Identify customer tone
3. Create English retrieval query for BM25


Categories:

- returns
- delivery
- payment
- account
- technical
- offers
- others


Tone:

- Strict
- Friendly
- Neutral



Complaint:

{query}



Return ONLY JSON.


Example:


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
                "role":"system",
                "content":
                "You classify customer complaints."
            },

            {
                "role":"user",
                "content":prompt
            }

        ]

    )



    try:

        content = (
            response
            .choices[0]
            .message
            .content
        )


        content = (
            content
            .replace("```json","")
            .replace("```","")
            .strip()
        )


        return json.loads(content)



    except:


        return {

            "retrieval_query":query,

            "category":"others",

            "tone":"Neutral"

        }





# ============================================================
# FILTER DATASET
# ============================================================


def filter_dataset_by_category(
        dataframe,
        category
):


    filtered = dataframe[

        dataframe["Category"]
        .astype(str)
        .str.lower()
        ==
        category.lower()

    ]



    if len(filtered)==0:

        return dataframe



    return filtered





# ============================================================
# BM25 INDEX
# ============================================================



def create_bm25_index(filtered_df):


    corpus=[]



    for _,row in filtered_df.iterrows():

        text=f"""

        {row['Trouble']}

        {row['Solution']}

        {row['Alternate Solution']}

        """

        corpus.append(text)



    tokenized=[

        tokenize_text(doc)

        for doc in corpus

    ]



    return BM25Okapi(
        tokenized
    )





# ============================================================
# RETRIEVAL
# ============================================================


def retrieve_documents(

        query,

        bm25,

        filtered_df,

        top_k=3

):


    tokens = tokenize_text(query)


    scores = bm25.get_scores(
        tokens
    )



    top_indices = np.argsort(scores)[

        -min(
            top_k,
            len(filtered_df)
        ):

    ][::-1]



    filtered_df = (
        filtered_df
        .reset_index(drop=True)
    )


    results=[]



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





# ============================================================
# SARVAM RESPONSE GENERATION
# ============================================================


def generate_customer_response(

        query,

        retrieved_docs,

        tone,

        language

):


    context=""



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


--------------------------------

"""




    prompt=f"""

You are a customer support executive.


Customer Complaint:

{query}



Tone:

{tone}



Knowledge Base:

{context}



Language:

{language}



Rules:

- Reply only in customer's language
- Do not reveal personal information
- Use only knowledge base
- Do not hallucinate



"""




    url="https://api.sarvam.ai/v1/chat/completions"



    headers={

        "api-subscription-key":
            SARVAM_API_KEY,

        "Content-Type":
            "application/json"

    }



    payload={

        "model":
            "sarvam-105b",


        "messages":[

            {
                "role":"system",

                "content":
                "Generate customer support replies."
            },


            {
                "role":"user",

                "content":prompt
            }

        ]

    }




    response=requests.post(

        url,

        headers=headers,

        json=payload

    )



    if response.status_code != 200:


        return (
            "Unable to generate response."
        )



    result=response.json()



    return (

        result["choices"][0]
        ["message"]
        ["content"]

    )







# ============================================================
# RAGAS VALIDATION
# ============================================================

def evaluate_rag_response(
        query,
        answer,
        retrieved_docs,
        ground_truth=None
):


    contexts = []


    for doc in retrieved_docs:

        context = f"""
Trouble:
{str(doc['trouble'])}

Solution:
{str(doc['solution'])}

Alternate Solution:
{str(doc['alternate_solution'])}

Company Response:
{str(doc['company_response'])}
"""


        contexts.append(context)



    evaluation_dict = {

        "user_input": [
            str(query)
        ],

        "response": [
            str(answer)
        ],

        "retrieved_contexts": [
            contexts
        ]

    }



    if ground_truth is not None:

        evaluation_dict["reference"] = [
            str(ground_truth)
        ]



    dataset = Dataset.from_dict(
        evaluation_dict
    )



    metrics = [

        metric_faithfulness,

        metric_answer_relevancy

    ]


    if ground_truth is not None:

        metrics.extend([

            metric_answer_correctness,

            metric_context_recall,

            metric_context_precision

        ])



    result = evaluate(

        dataset,

        metrics=metrics

    )


    return result

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# ============================================================
# UI
# ============================================================


st.title(
    "Customer Support Assistant"
)


st.write(
    "Voice + Guardrails + BM25 + OpenAI Classification + Sarvam + RAGAS Evaluation"
)




# ============================================================
# VOICE INPUT
# ============================================================


st.subheader(
    "Voice Complaint"
)



audio_file = st.audio_input(
    "Record your complaint"
)



if audio_file is not None:


    st.audio(
        audio_file
    )



    if st.button(
        "Transcribe Voice"
    ):


        with st.spinner(
            "Transcribing..."
        ):


            with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=".wav"

            ) as tmp:


                tmp.write(
                    audio_file.getvalue()
                )


                audio_path = tmp.name




            with open(
                audio_path,
                "rb"
            ) as f:


                response = (
                    sarvam_client
                    .speech_to_text
                    .transcribe(

                        file=f,

                        model="saarika:v2.5"

                    )
                )



            st.session_state.voice_query = (
                response.transcript
            )



            st.session_state.detected_language = (
                response.language_code
            )



            st.success(
                "Transcription complete"
            )



            st.write(
                "Transcript:"
            )


            st.write(
                response.transcript
            )


            st.write(
                "Language:",
                response.language_code
            )





# ============================================================
# TEXT INPUT
# ============================================================


text_query = st.text_area(
    "Or type your complaint"
)



query = (

    st.session_state.voice_query

    if st.session_state.voice_query

    else text_query

)




# ============================================================
# PROCESS QUERY
# ============================================================



if query:


    language = (
        st.session_state
        .detected_language
    )



    # ----------------------------
    # PII MASKING
    # ----------------------------


    pii_found, masked_query = mask_pii(
        query
    )



    if pii_found:


        st.subheader(
            "Masked Query"
        )


        st.write(
            masked_query
        )



    else:

        masked_query=query





    # ----------------------------
    # GUARDRAILS
    # ----------------------------


    valid, message = (
        run_input_guardrails(
            masked_query
        )
    )



    if not valid:


        st.error(
            message
        )

        st.stop()





    # ----------------------------
    # CACHE CHECK
    # ----------------------------


    cache_hit, cache_answer = (
        check_query_cache(
            masked_query
        )
    )



    if cache_hit:


        st.subheader(
            "Cached Answer"
        )


        st.write(
            cache_answer
        )


        st.stop()





    # ========================================================
    # RAG PIPELINE
    # ========================================================


    with st.spinner(
        "Generating response..."
    ):


        # ------------------------
        # Classification
        # ------------------------


        classification = classify_query(
            masked_query
        )



        retrieval_query = (
            classification["retrieval_query"]
        )


        category = (
            classification["category"]
        )


        tone = (
            classification["tone"]
        )




        st.write(
            "Category:",
            category
        )


        st.write(
            "Tone:",
            tone
        )




        # ------------------------
        # Retrieval
        # ------------------------


        filtered_df = (
            filter_dataset_by_category(
                df,
                category
            )
        )



        bm25 = create_bm25_index(
            filtered_df
        )



        retrieved_docs = retrieve_documents(

            retrieval_query,

            bm25,

            filtered_df,

            top_k=3

        )




        if not retrieved_docs:


            st.warning(
                "No relevant documents found."
            )

            st.stop()





        # ------------------------
        # Generation
        # ------------------------


        final_answer = (
            generate_customer_response(

                masked_query,

                retrieved_docs,

                tone,

                language

            )
        )






    # ========================================================
    # FIND GROUND TRUTH
    # ========================================================


    ground_truth = None


    ragas_df = ragas_testset.to_pandas()


    for _, row in ragas_df.iterrows():

        if row["user_input"].lower() in masked_query.lower():

            ground_truth = row["reference"]

            break






    # ========================================================
    # RAGAS VALIDATION
    # ========================================================


    st.subheader(
        "RAGAS Evaluation"
    )



    with st.spinner(
        "Evaluating answer..."
    ):


        try:


            ragas_result = evaluate_rag_response(

                masked_query,

                final_answer,

                retrieved_docs,

                ground_truth

            )



            st.dataframe(
                ragas_result.to_pandas()
            )



        except Exception as e:


            st.error(
                f"RAGAS evaluation failed: {e}"
            )







    # ========================================================
    # SAVE CACHE
    # ========================================================


    save_query_cache(

        masked_query,

        None,

        final_answer

    )






    # ========================================================
    # FINAL RESPONSE
    # ========================================================


    st.subheader(
        "Answer"
    )


    st.write(
        final_answer
    )





    # ========================================================
    # RETRIEVED DOCUMENTS
    # ========================================================


    st.subheader(
        "Retrieved Knowledge"
    )



    for i,doc in enumerate(

        retrieved_docs,

        start=1

    ):


        with st.expander(
            f"Result {i}"
        ):



            st.write(

                "Score:",

                round(
                    doc["score"],
                    3
                )

            )



            st.write(

                "Trouble:",

                doc["trouble"]

            )


            st.write(

                "Solution:",

                doc["solution"]

            )


            st.write(

                "Alternate Solution:",

                doc["alternate_solution"]

            )


            st.write(

                "Company Response:",

                doc["company_response"]

            )