import os
import re
import streamlit as st
import numpy as np
import chromadb

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# -------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not loaded. Please check your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)
chroma_client = chromadb.Client()

# -------------------------------
# 2. PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Multi PDF RAG", page_icon="📚", layout="wide")

st.title("📚 Multi-PDF RAG - Rule-Based Metadata Filtering")
st.write("Upload multiple PDFs and ask questions. The application automatically selects a document category using predefined rules.")

# -------------------------------
# 3. CATEGORY RULES
# -------------------------------
CATEGORY_RULES = {
    "HR": ["leave","salary","employee","attendance","holiday","recruitment","performance","promotion","benefits","payroll"],
    "Finance": ["expense","reimbursement","budget","invoice","payment","tax","finance","financial","accounts","purchase"],
    "IT": ["software","hardware","password","security","network","laptop","system","vpn","database","technical"],
    "Legal": ["contract","agreement","compliance","law","legal","policy","regulation","clause","copyright","license"]
}

# -------------------------------
# 4. CATEGORY DETECTION
# -------------------------------
def detect_category_rule_based(query):
    query = query.lower()
    category_scores = {cat: sum(1 for kw in kws if kw in query) for cat, kws in CATEGORY_RULES.items()}
    best_category = max(category_scores, key=category_scores.get)
    return best_category if category_scores[best_category] > 0 else "All"

# -------------------------------
# 5. PDF TEXT EXTRACTION
# -------------------------------
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pages_text = []
    for page_number, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            pages_text.append({"page": page_number + 1, "text": text})
    return pages_text

# -------------------------------
# 6. CHUNKING
# -------------------------------
def create_chunks(pages_text, filename, category, chunk_size=500, chunk_overlap=100):
    chunks = []
    for page in pages_text:
        text = page["text"]
        page_number = page["page"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "id": len(chunks),
                    "text": chunk_text,
                    "page": page_number,
                    "filename": filename,
                    "category": category
                })
            start = end - chunk_overlap
    return chunks

# -------------------------------
# 7. EMBEDDINGS
# -------------------------------
def create_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

# -------------------------------
# 8. CHROMADB COLLECTION
# -------------------------------
def create_chroma_collection(chunks, collection_name="rag_demo"):
    collection = chroma_client.get_or_create_collection(name=collection_name)    
    for chunk in chunks:
        embedding = create_embedding(chunk["text"])
        collection.add(
            ids=[str(chunk["id"])],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "page": chunk["page"],
                "filename": chunk["filename"],
                "category": chunk["category"]
            }]
        )
    return collection

# -------------------------------
# 9. BM25 INDEX
# -------------------------------
def tokenize_text(text):
    return re.findall(r"\b\w+\b", text.lower())

def create_bm25_index(chunks):
    tokenized_documents = [tokenize_text(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_documents)

# -------------------------------
# 10. FILTER CHUNKS
# -------------------------------
def filter_chunks_by_category(chunks, category):
    return chunks if category == "All" else [chunk for chunk in chunks if chunk["category"] == category]

# -------------------------------
# 11. SEMANTIC SEARCH (ChromaDB)
# -------------------------------
def semantic_retrieval(query, collection, top_k=10):
    query_embedding = create_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
            "filename": results["metadatas"][0][i]["filename"],
            "category": results["metadatas"][0][i]["category"],
            "distance": None
        })
    return retrieved

# -------------------------------
# 12. BM25 SEARCH
# -------------------------------
def keyword_retrieval(query, bm25, chunks, top_k=10):
    tokenized_query = tokenize_text(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-min(top_k, len(chunks)):][::-1]
    results = []
    for index in top_indices:
        results.append({
            "chunk_id": chunks[index]["id"],
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "filename": chunks[index]["filename"],
            "category": chunks[index]["category"],
            "bm25_score": float(scores[index])
        })
    return results

# -------------------------------
# 13. RRF
# -------------------------------
def reciprocal_rank_fusion(semantic_results, keyword_results, k=60):
    rrf_scores, chunk_data = {}, {}
    for rank, result in enumerate(semantic_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1/(k+rank+1)
        chunk_data[cid] = result
    for rank, result in enumerate(keyword_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1/(k+rank+1)
        chunk_data[cid] = result
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final_results = []
    for cid, score in sorted_chunks:
        result = chunk_data[cid].copy()
        result["rrf_score"] = score
        final_results.append(result)
    return final_results

# -------------------------------
# 14. GENERATE ANSWER
# -------------------------------
def generate_answer(query, retrieved_chunks):
    context = ""
    for chunk in retrieved_chunks:
        context += f"""
Source:
Document: {chunk["filename"]}
Category: {chunk["category"]}
Page: {chunk["page"]}

Content:
{chunk["text"]}
"""
    prompt = f"""
You are a helpful document question-answering assistant.
Answer the user's question using ONLY the information in the provided context.
Do not use outside knowledge.
If the answer cannot be found in the context, say:
"I could not find the answer in the uploaded documents."
Always mention the source document and page number when possible.

Context:
{context}

User Question:
{query}

Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the retrieved document context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# -------------------------------
# 15. STREAMLIT UI
# -------------------------------
uploaded_files = st.file_uploader("Upload multiple PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    with st.spinner("Processing multiple PDF documents..."):
        for uploaded_file in uploaded_files:
            pages_text = extract_text_from_pdf(uploaded_file)
            filename_lower = uploaded_file.name.lower()
            if any(kw in filename_lower for kw in CATEGORY_RULES["HR"]): category = "HR"
            elif any(kw in filename_lower for kw in CATEGORY_RULES["Finance"]): category = "Finance"
            elif any(kw in filename_lower for kw in CATEGORY_RULES["IT"]): category = "IT"
            elif any(kw in filename_lower for kw in CATEGORY_RULES["Legal"]): category = "Legal"
            else: category = "Other"
            pdf_chunks = create_chunks(pages_text, uploaded_file.name, category)
            all_chunks.extend(pdf_chunks)

    st.success(f"Processed {len(uploaded_files)} PDFs and created {len(all_chunks)} chunks.")

    st.subheader("Document Metadata")
    for filename in set(chunk["filename"] for chunk in all_chunks):
        file_chunks = [chunk for chunk in all_chunks if chunk["filename"] == filename]
        st.write(f"📄 {filename} | Category: {file_chunks[0]['category']}")

    # -------------------------------
    # Query Input
    # -------------------------------
    query = st.text_input("Ask a question about your documents:")

    if query:
        # Detect category
        detected_category = detect_category_rule_based(query)
        st.info(f"Rule-based category detected: **{detected_category}**")

        # Filter chunks
        filtered_chunks = filter_chunks_by_category(all_chunks, detected_category)
        if not filtered_chunks:
            st.warning("No documents found for this category.")
            st.stop()

        st.write(f"Searching across {len(filtered_chunks)} filtered chunks.")

        # Create indexes
        with st.spinner("Creating filtered search indexes..."):
            filtered_collection = create_chroma_collection(filtered_chunks, "filtered_rag_demo")
            filtered_bm25 = create_bm25_index(filtered_chunks)

        # Semantic Retrieval
        semantic_results = semantic_retrieval(query, filtered_collection, top_k=10)

        # BM25 Retrieval
        keyword_results = keyword_retrieval(query, filtered_bm25, filtered_chunks, top_k=10)

        # Reciprocal Rank Fusion
        reranked_results = reciprocal_rank_fusion(semantic_results, keyword_results)
        final_results = reranked_results[:3]

        # Generate Answer
        answer = generate_answer(query, final_results)

        # Display Answer
        st.subheader("Answer")
        st.write(answer)

            # -------------------------------
        # Display Sources
        # -------------------------------
        st.subheader("Retrieved Sources")
        for i, chunk in enumerate(final_results):
            with st.expander(f"Result {i+1} | {chunk['filename']} | Page {chunk['page']}"):
                st.write(chunk["text"])
                st.write(f"Category: {chunk['category']}")
                st.write(f"RRF Score: {chunk['rrf_score']:.6f}")