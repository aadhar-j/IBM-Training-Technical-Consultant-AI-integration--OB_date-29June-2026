import os
import re
import uuid
import streamlit as st
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from pinecone import Pinecone, ServerlessSpec

# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found")
    st.stop()

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not found")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Multi PDF Hybrid RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF Hybrid RAG (Pinecone + BM25 + RRF)")
st.write("Upload multiple PDFs and ask questions.")

# --------------------------------------------------
# PINECONE SETUP
# --------------------------------------------------

INDEX_NAME = "multi-pdf-rag"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

# --------------------------------------------------
# CATEGORY RULES
# --------------------------------------------------

CATEGORY_RULES = {
    "HR": [
        "leave",
        "salary",
        "employee",
        "attendance",
        "holiday",
        "recruitment",
        "performance",
        "promotion",
        "benefits",
        "payroll"
    ],
    "Finance": [
        "expense",
        "reimbursement",
        "budget",
        "invoice",
        "payment",
        "tax",
        "finance",
        "financial",
        "accounts",
        "purchase"
    ],
    "IT": [
        "software",
        "hardware",
        "password",
        "security",
        "network",
        "laptop",
        "system",
        "vpn",
        "database",
        "technical"
    ],
    "Legal": [
        "contract",
        "agreement",
        "compliance",
        "law",
        "legal",
        "policy",
        "regulation",
        "clause",
        "copyright",
        "license"
    ]
}

# --------------------------------------------------
# CATEGORY DETECTION
# --------------------------------------------------

def detect_category_rule_based(query):

    query = query.lower()

    scores = {
        category: sum(
            1 for keyword in keywords
            if keyword in query
        )
        for category, keywords in CATEGORY_RULES.items()
    }

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "All"

    return best

# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    pages = []

    for page_num, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:
            pages.append({
                "page": page_num + 1,
                "text": text
            })

    return pages

# --------------------------------------------------
# CHUNKING
# --------------------------------------------------

def create_chunks(
    pages_text,
    filename,
    category,
    chunk_size=500,
    chunk_overlap=100
):

    chunks = []

    for page in pages_text:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            if chunk.strip():

                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "page": page_number,
                    "filename": filename,
                    "category": category
                })

            start += chunk_size - chunk_overlap

    return chunks

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

# --------------------------------------------------
# STORE IN PINECONE
# --------------------------------------------------

def store_chunks_in_pinecone(chunks):

    vectors = []

    for chunk in chunks:

        embedding = create_embedding(chunk["text"])

        vectors.append({
            "id": chunk["id"],
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "filename": chunk["filename"],
                "category": chunk["category"]
            }
        })

    batch_size = 100

    for i in range(0, len(vectors), batch_size):
        index.upsert(
            vectors=vectors[i:i+batch_size]
        )

# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------

def tokenize_text(text):
    return re.findall(
        r"\b\w+\b",
        text.lower()
    )

# --------------------------------------------------
# BM25
# --------------------------------------------------

def create_bm25_index(chunks):

    tokenized_docs = [
        tokenize_text(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_docs)

# --------------------------------------------------
# FILTER CHUNKS
# --------------------------------------------------

def filter_chunks_by_category(
    chunks,
    category
):

    if category == "All":
        return chunks

    return [
        chunk
        for chunk in chunks
        if chunk["category"] == category
    ]

# --------------------------------------------------
# PINECONE RETRIEVAL
# --------------------------------------------------

def semantic_retrieval(
    query,
    category,
    top_k=10
):

    query_embedding = create_embedding(query)

    pinecone_filter = {}

    if category != "All":
        pinecone_filter = {
            "category": {
                "$eq": category
            }
        }

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter
    )

    retrieved = []

    for match in results["matches"]:

        retrieved.append({
            "chunk_id": match["id"],
            "text": match["metadata"]["text"],
            "page": match["metadata"]["page"],
            "filename": match["metadata"]["filename"],
            "category": match["metadata"]["category"],
            "score": match["score"]
        })

    return retrieved

# --------------------------------------------------
# BM25 RETRIEVAL
# --------------------------------------------------

def keyword_retrieval(
    query,
    bm25,
    chunks,
    top_k=10
):

    query_tokens = tokenize_text(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[
        -min(top_k, len(chunks)):
    ][::-1]

    results = []

    for idx in top_indices:

        results.append({
            "chunk_id": chunks[idx]["id"],
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "filename": chunks[idx]["filename"],
            "category": chunks[idx]["category"],
            "bm25_score": float(scores[idx])
        })

    return results

# --------------------------------------------------
# RRF
# --------------------------------------------------

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k=60
):

    rrf_scores = {}
    chunk_map = {}

    for rank, result in enumerate(semantic_results):

        cid = result["chunk_id"]

        rrf_scores[cid] = (
            rrf_scores.get(cid, 0)
            + 1 / (k + rank + 1)
        )

        chunk_map[cid] = result

    for rank, result in enumerate(keyword_results):

        cid = result["chunk_id"]

        rrf_scores[cid] = (
            rrf_scores.get(cid, 0)
            + 1 / (k + rank + 1)
        )

        chunk_map[cid] = result

    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    final_results = []

    for cid, score in sorted_results:

        item = chunk_map[cid].copy()
        item["rrf_score"] = score

        final_results.append(item)

    return final_results

# --------------------------------------------------
# ANSWER GENERATION
# --------------------------------------------------

def generate_answer(
    query,
    retrieved_chunks
):

    context = ""

    for chunk in retrieved_chunks:

        context += f"""
Document: {chunk['filename']}
Category: {chunk['category']}
Page: {chunk['page']}

Content:
{chunk['text']}

"""

    prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{query}

If answer is unavailable say:
"I could not find the answer in the uploaded documents."
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

# --------------------------------------------------
# UI
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []

    with st.spinner("Processing PDFs..."):

        for uploaded_file in uploaded_files:

            pages = extract_text_from_pdf(
                uploaded_file
            )

            filename_lower = uploaded_file.name.lower()

            if any(
                kw in filename_lower
                for kw in CATEGORY_RULES["HR"]
            ):
                category = "HR"

            elif any(
                kw in filename_lower
                for kw in CATEGORY_RULES["Finance"]
            ):
                category = "Finance"

            elif any(
                kw in filename_lower
                for kw in CATEGORY_RULES["IT"]
            ):
                category = "IT"

            elif any(
                kw in filename_lower
                for kw in CATEGORY_RULES["Legal"]
            ):
                category = "Legal"

            else:
                category = "Other"

            chunks = create_chunks(
                pages,
                uploaded_file.name,
                category
            )

            all_chunks.extend(chunks)

    st.success(
        f"{len(all_chunks)} chunks created"
    )

    if st.button("Store in Pinecone"):
        with st.spinner("Uploading embeddings..."):
            store_chunks_in_pinecone(all_chunks)

        st.success("Stored in Pinecone")

    query = st.text_input(
        "Ask a question"
    )

    if query:

        detected_category = detect_category_rule_based(query)

        st.info(
            f"Detected Category: {detected_category}"
        )

        filtered_chunks = filter_chunks_by_category(
            all_chunks,
            detected_category
        )

        bm25 = create_bm25_index(
            filtered_chunks
        )

        semantic_results = semantic_retrieval(
            query,
            detected_category,
            top_k=10
        )

        keyword_results = keyword_retrieval(
            query,
            bm25,
            filtered_chunks,
            top_k=10
        )

        fused_results = reciprocal_rank_fusion(
            semantic_results,
            keyword_results
        )

        final_results = fused_results[:3]

        answer = generate_answer(
            query,
            final_results
        )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Sources")

        for i, chunk in enumerate(final_results):

            with st.expander(
                f"Result {i+1} | {chunk['filename']} | Page {chunk['page']}"
            ):
                st.write(chunk["text"])

                st.write(
                    f"Category: {chunk['category']}"
                )

                st.write(
                    f"RRF Score: {chunk['rrf_score']:.6f}"
                )