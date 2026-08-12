import os
import streamlit as st

from dotenv import load_dotenv

from openai import OpenAI
import numpy as np
from pypdf import PdfReader

import chromadb


# ============================
# RAGAS IMPORTS
# ============================

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    LLMContextRecall,
    LLMContextPrecisionWithReference
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.testset import TestsetGenerator

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)

from langchain_core.documents import Document



# ============================
# SETUP
# ============================

load_dotenv()


api_key = os.getenv(
    "OPENAI_API_KEY"
)


if not api_key:

    st.error(
        "OPENAI_API_KEY missing"
    )

    st.stop()



client = OpenAI(
    api_key=api_key
)



st.set_page_config(

    page_title="Employee Handbook RAG",

    layout="wide"

)



st.title(
    "Employee Handbook RAG + RAGAS Evaluation"
)



# ============================
# MODELS
# ============================


llm = ChatOpenAI(

    model="gpt-4o-mini",

    temperature=0

)



embeddings = OpenAIEmbeddings(

    model="text-embedding-3-small"

)



ragas_llm = LangchainLLMWrapper(
    llm
)


ragas_embeddings = LangchainEmbeddingsWrapper(
    embeddings
)



# ============================
# RAGAS METRICS
# ============================


faithfulness = Faithfulness(
    llm=ragas_llm
)


answer_relevancy = AnswerRelevancy(

    llm=ragas_llm,

    embeddings=ragas_embeddings

)


answer_correctness = AnswerCorrectness(

    llm=ragas_llm,

    embeddings=ragas_embeddings

)


context_recall = LLMContextRecall(

    llm=ragas_llm

)


context_precision = LLMContextPrecisionWithReference(

    llm=ragas_llm

)





# ============================
# PDF LOADING
# ============================


def load_pdf(file):

    reader = PdfReader(file)

    text=""


    for page in reader.pages:

        text += (
            page.extract_text()
            + "\n"
        )


    return text




# ============================
# CHUNKING
# ============================


def chunk_text(
        text,
        chunk_size=700,
        overlap=100
):

    chunks=[]


    start=0


    while start < len(text):


        end=start+chunk_size


        chunks.append(
            text[start:end]
        )


        start += (
            chunk_size-overlap
        )


    return chunks





# ============================
# EMBEDDINGS
# ============================


def get_embedding(text):


    response = client.embeddings.create(

        model="text-embedding-3-small",

        input=text

    )


    return response.data[0].embedding





# ============================
# CHROMA DB
# ============================


@st.cache_resource
def create_vector_db(chunks):


    chroma_client = chromadb.Client()



    collection = chroma_client.create_collection(

        name="employee_handbook"

    )



    for i,chunk in enumerate(chunks):


        collection.add(

            ids=[str(i)],

            embeddings=[

                get_embedding(chunk)

            ],

            documents=[chunk]

        )


    return collection





# ============================
# RETRIEVAL
# ============================


def retrieve(

        query,

        collection,

        top_k=3

):


    query_embedding = get_embedding(
        query
    )



    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=top_k

    )


    return results["documents"][0]





# ============================
# GENERATION
# ============================


def generate_answer(

        query,

        contexts

):


    context="\n\n".join(
        contexts
    )


    prompt=f"""

You are an employee handbook assistant.

Answer only using the provided context.

If answer is not available,
say you don't know.


Context:

{context}



Question:

{query}

"""


    response = client.chat.completions.create(

        model="gpt-4o-mini",

        temperature=0,

        messages=[

            {
                "role":"user",

                "content":prompt
            }

        ]

    )


    return (
        response
        .choices[0]
        .message
        .content
    )





# ============================
# CREATE SYNTHETIC GROUND TRUTH
# ============================


@st.cache_resource
def create_ragas_testset(chunks):


    documents=[]


    combined=""


    for chunk in chunks:

        combined += chunk + "\n\n"



    documents.append(

        Document(

            page_content=combined

        )

    )



    generator = TestsetGenerator.from_langchain(

        llm,

        embeddings

    )



    testset = generator.generate_with_langchain_docs(

        documents,

        testset_size=10

    )



    return testset





# ============================
# RAGAS EVALUATION
# ============================


def evaluate_response(

        query,

        answer,

        contexts,

        reference

):


    data={


        "user_input":[query],


        "response":[answer],


        "retrieved_contexts":[contexts],


        "reference":[reference]


    }



    dataset = Dataset.from_dict(
        data
    )



    result = evaluate(

        dataset,

        metrics=[

            faithfulness,

            answer_relevancy,

            answer_correctness,

            context_recall,

            context_precision

        ]

    )


    return result





# ============================
# STREAMLIT UI
# ============================


uploaded_file = st.file_uploader(

    "Upload Employee Handbook PDF",

    type=["pdf"]

)



if uploaded_file:


    text = load_pdf(
        uploaded_file
    )


    chunks = chunk_text(
        text
    )



    st.success(
        f"{len(chunks)} chunks created"
    )



    collection=create_vector_db(
        chunks
    )



    ragas_testset=create_ragas_testset(
        chunks
    )



    query = st.text_input(
        "Ask your question"
    )



    if query:


        retrieved_chunks = retrieve(

            query,

            collection

        )



        answer = generate_answer(

            query,

            retrieved_chunks

        )



        st.subheader(
            "Answer"
        )

        st.write(
            answer
        )



        # -----------------------
        # Find synthetic reference
        # -----------------------


        reference = None


        test_df = ragas_testset.to_pandas()


        # semantic matching using embeddings

        query_embedding = get_embedding(query)


        best_score = 0


        for _, row in test_df.iterrows():


            synthetic_question = row["user_input"]


            synthetic_embedding = get_embedding(
                synthetic_question
            )


            similarity = np.dot(
                query_embedding,
                synthetic_embedding
            ) / (
                np.linalg.norm(query_embedding)
                *
                np.linalg.norm(synthetic_embedding)
            )



            if similarity > best_score:

                best_score = similarity

                reference = row["reference"]



        if best_score < 0.80:

            reference = None





        if reference:


            score=evaluate_response(

                query,

                answer,

                retrieved_chunks,

                reference

            )


            st.subheader(
                "RAGAS Scores"
            )


            st.dataframe(
                score.to_pandas()
            )


        else:


            st.warning(
                "No synthetic ground truth found for this question."
            )



        with st.expander(
            "Retrieved Context"
        ):


            for i,c in enumerate(
                retrieved_chunks,
                1
            ):

                st.write(
                    f"Chunk {i}"
                )

                st.write(c)

                st.divider()