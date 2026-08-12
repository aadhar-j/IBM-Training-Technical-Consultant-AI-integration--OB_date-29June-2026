import os
import json
import re

import streamlit as st
import chromadb
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from datasets import Dataset

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)

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

from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

if not OPENAI_API_KEY:

    st.error(
        "OPENAI_API_KEY missing"
    )

    st.stop()

client = OpenAI(
    api_key=OPENAI_API_KEY
)

st.set_page_config(
    page_title="Employee Handbook RAG",
    layout="wide"
)

st.title(
    "Employee Handbook RAG + GPT-4 Evaluation"
)


# -----------------------------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------------------------

generation_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

evaluation_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

ragas_llm = LangchainLLMWrapper(
    evaluation_llm
)

ragas_embeddings = (
    LangchainEmbeddingsWrapper(
        embeddings
    )
)

# -----------------------------------------------------------------------------------------------
# RAGAS METRICS
# -----------------------------------------------------------------------------------------------

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

context_precision = (
    LLMContextPrecisionWithReference(
        llm=ragas_llm
    )
)

# -----------------------------------------------------------------------------------------------
# SEMANTIC CACHE
# -----------------------------------------------------------------------------------------------

QUERY_CACHE = []

CACHE_SIMILARITY_THRESHOLD = 0.95

def check_query_cache(query):

    query_embedding = embeddings.embed_query(
        query
    )

    for item in QUERY_CACHE:

        similarity = cosine_similarity(
            [query_embedding],
            [item["embedding"]]
        )[0][0]

        if similarity >= CACHE_SIMILARITY_THRESHOLD:

            return True, item["answer"]

    return False, query_embedding

def save_query_cache(
    query,
    embedding,
    answer
):

    QUERY_CACHE.append(
        {
            "question": query,
            "embedding": embedding,
            "answer": answer
        }
    )

# -----------------------------------------------------------------------------------------------
# GUARDRAILS
# -----------------------------------------------------------------------------------------------

DOMAIN_DESCRIPTION = [

    "employee handbook",
    "leave policy",
    "annual leave",
    "casual leave",
    "sick leave",
    "maternity leave",
    "paternity leave",

    "attendance policy",
    "working hours",

    "probation period",

    "employee benefits",

    "insurance",

    "travel policy",

    "expense reimbursement",

    "code of conduct",

    "disciplinary action",

    "work from home",

    "remote work",

    "holiday policy",

    "notice period",

    "termination policy",

    "promotion policy",

    "performance review"
]

@st.cache_resource
def build_domain_embeddings():

    return [

        embeddings.embed_query(item)

        for item in DOMAIN_DESCRIPTION

    ]

domain_embeddings = build_domain_embeddings()


def semantic_domain_validation(
    question
):

    q_emb = embeddings.embed_query(
        question
    )

    scores = [

        cosine_similarity(
            [q_emb],
            [d]
        )[0][0]

        for d in domain_embeddings

    ]

    if max(scores) >= 0.40:

        return True, ""

    return (
        False,
        "Question appears outside handbook domain."
    )



EMAIL_PATTERN = r"\S+@\S+\.\S+"

PHONE_PATTERN = r"\b\d{10}\b"

AADHAR_PATTERN = r"\b\d{4}\s?\d{4}\s?\d{4}\b"

PAN_PATTERN = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"


def mask_pii(question):

    original = question

    question = re.sub(
        EMAIL_PATTERN,
        "[EMAIL]",
        question
    )

    question = re.sub(
        PHONE_PATTERN,
        "[PHONE]",
        question
    )

    question = re.sub(
        AADHAR_PATTERN,
        "[AADHAR]",
        question
    )

    question = re.sub(
        PAN_PATTERN,
        "[PAN]",
        question
    )

    return (
        original != question,
        question
    )


# -----------------------------------------------------------------------------------------------
# LOAD PDF
# -----------------------------------------------------------------------------------------------

def load_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# -----------------------------------------------------------------------------------------------
# CHUNKING
# -----------------------------------------------------------------------------------------------

def chunk_text(
    text,
    chunk_size=1000,
    overlap=150
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += (
            chunk_size - overlap
        )

    return chunks


# -----------------------------------------------------------------------------------------------
# EMBEDDINGS
# -----------------------------------------------------------------------------------------------

def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

# -----------------------------------------------------------------------------------------------
# CHROMADB
# -----------------------------------------------------------------------------------------------

@st.cache_resource
def create_vector_db(chunks):

    chroma_client = chromadb.Client()

    try:

        chroma_client.delete_collection(
            "employee_handbook"
        )

    except:
        pass

    collection = (
        chroma_client.create_collection(
            name="employee_handbook"
        )
    )

    for i, chunk in enumerate(chunks):

        collection.add(
            ids=[str(i)],
            embeddings=[
                get_embedding(chunk)
            ],
            documents=[chunk]
        )

    return collection


# -----------------------------------------------------------------------------------------------
# RETERIVAL
# -----------------------------------------------------------------------------------------------

def retrieve(
    query,
    collection,
    top_k=3
):

    query_embedding = get_embedding(
        query
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k
    )

    return results["documents"][0]


# -----------------------------------------------------------------------------------------------
# ANSWER GENERATION
# -----------------------------------------------------------------------------------------------

def generate_answer(
    query,
    contexts
):

    context = "\n\n".join(
        contexts
    )

    prompt = f"""
You are an Employee Handbook Assistant.

Answer ONLY using the information
contained in the handbook context.

If the answer is unavailable,
say:

"I could not find this information
in the handbook."

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
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# -----------------------------------------------------------------------------------------------
# GENERATE SYNTHETIC Q/A PAIRS
# -----------------------------------------------------------------------------------------------

def generate_qa_pairs(
    chunk
):

    prompt = f"""
You are generating a RAG evaluation dataset.

Using ONLY the handbook content below,
generate 3 question-answer pairs.

Return valid JSON only.

Example:

[
  {{
    "question":"What is the probation period?",
    "answer":"The probation period is six months."
  }}
]

HANDBOOK CONTENT:

{chunk}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    content = (
        content
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    return json.loads(
        content
    )


# -----------------------------------------------------------------------------------------------
# SAVE SYNTHETIC Q/A PAIRS IF NOT PRESENT ELSE LOAD JSON
# -----------------------------------------------------------------------------------------------


@st.cache_resource
def create_evaluation_dataset(
    chunks
):

    filename = (
        "employee_handbook_eval.json"
    )

    if os.path.exists(
        filename
    ):

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    dataset = []

    progress = st.progress(0)

    total = len(chunks)

    for i, chunk in enumerate(chunks):

        try:

            qa_pairs = (
                generate_qa_pairs(
                    chunk
                )
            )

            dataset.extend(
                qa_pairs
            )

        except Exception as e:

            st.error(
            f"Q&A Generation Failed: {e}"
            )

            raise e

        progress.progress(
            (i + 1) / total
        )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    return dataset


# -----------------------------------------------------------------------------------------------
# EVALUATION EMBEDDINGS
# -----------------------------------------------------------------------------------------------


@st.cache_resource
def build_evaluation_index(
    evaluation_data
):

    questions = []

    answers = []

    embeddings_list = []

    for item in evaluation_data:

        questions.append(
            item["question"]
        )

        answers.append(
            item["answer"]
        )

        embeddings_list.append(
            get_embedding(
                item["question"]
            )
        )

    return (
        questions,
        answers,
        np.array(
            embeddings_list
        )
    )


# -----------------------------------------------------------------------------------------------
# GROUND TRUTH RETRIVAL
# -----------------------------------------------------------------------------------------------

def find_reference_answer(
    query,
    eval_questions,
    eval_answers,
    eval_embeddings
):

    query_embedding = np.array(
        get_embedding(
            query
        )
    )

    similarities = []

    for emb in eval_embeddings:

        similarity = (
            np.dot(
                query_embedding,
                emb
            )
            /
            (
                np.linalg.norm(
                    query_embedding
                )
                *
                np.linalg.norm(
                    emb
                )
            )
        )

        similarities.append(
            similarity
        )

    best_idx = int(
        np.argmax(
            similarities
        )
    )

    return (
        eval_questions[
            best_idx
        ],
        eval_answers[
            best_idx
        ],
        similarities[
            best_idx
        ]
    )


# -----------------------------------------------------------------------------------------------
# RAGAS EVALUATION
# -----------------------------------------------------------------------------------------------

def evaluate_response(
    query,
    answer,
    contexts,
    reference
):

    dataset = Dataset.from_dict(
        {
            "user_input": [
                query
            ],

            "response": [
                answer
            ],

            "retrieved_contexts": [
                contexts
            ],

            "reference": [
                reference
            ]
        }
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


# ----------------------------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Employee Handbook PDF",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner(
        "Loading handbook..."
    ):

        text = load_pdf(
            uploaded_file
        )

        chunks = chunk_text(
            text
        )

    st.success(
        f"{len(chunks)} chunks created"
    )

    # -------------------------------------------------------
    # VECTOR DB
    # -------------------------------------------------------

    with st.spinner(
        "Building vector database..."
    ):

        collection = create_vector_db(
            chunks
        )

    # -------------------------------------------------------
    # SYNTHETIC EVALUATION DATASET
    # -------------------------------------------------------

    with st.spinner(
        "Loading evaluation dataset..."
    ):

        evaluation_data = (
            create_evaluation_dataset(
                chunks
            )
        )

        (
            eval_questions,
            eval_answers,
            eval_embeddings
        ) = build_evaluation_index(
            evaluation_data
        )

    st.success(
        f"{len(evaluation_data)} evaluation samples loaded"
    )

    # -------------------------------------------------------
    # USER QUESTION
    # -------------------------------------------------------

    query = st.text_input(
        "Ask a handbook question"
    )

    if query:

        # ---------------------------------------------------
        # PII MASKING
        # ---------------------------------------------------

        pii_found, masked_query = (
            mask_pii(
                query
            )
        )

        if pii_found:

            st.warning(
                "PII detected and masked."
            )

            st.write(
                masked_query
            )

        # ---------------------------------------------------
        # DOMAIN VALIDATION
        # ---------------------------------------------------

        valid, message = (
            semantic_domain_validation(
                masked_query
            )
        )

        if not valid:

            st.error(
                message
            )

            st.stop()

        # ---------------------------------------------------
        # CACHE CHECK
        # ---------------------------------------------------

        cache_hit, cache_data = (
            check_query_cache(
                masked_query
            )
        )

        if cache_hit:

            st.subheader(
                "Cached Answer"
            )

            st.write(
                cache_data
            )

            st.stop()

        question_embedding = (
            cache_data
        )

        # ---------------------------------------------------
        # RETRIEVAL
        # ---------------------------------------------------

        with st.spinner(
            "Retrieving handbook context..."
        ):

            retrieved_chunks = retrieve(
                masked_query,
                collection,
                top_k=3
            )

        # ---------------------------------------------------
        # ANSWER GENERATION
        # ---------------------------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                masked_query,
                retrieved_chunks
            )

        save_query_cache(
            masked_query,
            question_embedding,
            answer
        )

        st.subheader(
            "Answer"
        )

        st.write(
            answer
        )

        # ---------------------------------------------------
        # GROUND TRUTH MATCHING
        # ---------------------------------------------------

        (
            matched_question,
            reference_answer,
            similarity
        ) = find_reference_answer(
            masked_query,
            eval_questions,
            eval_answers,
            eval_embeddings
        )

        st.subheader(
            "Evaluation Reference"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "**Matched Question**"
            )

            st.write(
                matched_question
            )

        with col2:

            st.markdown(
                "**Similarity**"
            )

            st.write(
                f"{similarity:.3f}"
            )

        st.markdown(
            "**Ground Truth Answer**"
        )

        st.write(
            reference_answer
        )

        # ---------------------------------------------------
        # RAGAS EVALUATION
        # ---------------------------------------------------

        with st.spinner(
            "Evaluating response..."
        ):

            try:

                ragas_result = (
                    evaluate_response(
                        masked_query,
                        answer,
                        retrieved_chunks,
                        reference_answer
                    )
                )

                st.subheader(
                    "RAGAS Evaluation"
                )

                result_df = (
                    ragas_result.to_pandas()
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Evaluation failed: {e}"
                )

        # ---------------------------------------------------
        # RETRIEVED CONTEXT
        # ---------------------------------------------------

        with st.expander(
            "Retrieved Context"
        ):

            for i, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):

                st.markdown(
                    f"### Chunk {i}"
                )

                st.write(
                    chunk
                )

                st.divider()

        # ---------------------------------------------------
        # DEBUG INFO
        # ---------------------------------------------------

        with st.expander(
            "Debug Information"
        ):

            st.write(
                {
                    "query": masked_query,
                    "num_chunks":
                        len(chunks),
                    "retrieved_chunks":
                        len(
                            retrieved_chunks
                        ),
                    "evaluation_samples":
                        len(
                            evaluation_data
                        ),
                    "similarity":
                        float(
                            similarity
                        )
                }
            )