import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import faiss
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not loaded. Please check your .env file")
    st.stop()

client = OpenAI(
    api_key=api_key
)

# --------------------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="pdf",
    layout="wide"
)

st.title("PDF RAG Assistant")
st.write("Upload a PDF and ask questions about its content")

# ----------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF Document",
    type=["pdf"]
)

# ------------------------------------------------------------------

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pages_text = []

    for page_number, page in enumerate(pdf_reader.pages):
        text = page.extract_text()

        if text:
            pages_text.append({
                "page": page_number + 1,
                "text": text                
            })

    return pages_text


# -------------------------------------------------------------------

def create_chunks(pages_text, chunk_size=300, chunk_overlap=50):
    chunks = []
    for page in pages_text:
        text = page["text"]
        page_number = page["page"]
        start = 0

        while start < len(text):
                end = min(start+chunk_size, len(text))
                chunk_text = text[start:end]
                chunks.append({
                    'chunk_id' : len(chunks),
                    'page': page_number,
                    'text': chunk_text,
                })
        
                start = end-chunk_overlap       
                if end >= len(text)-1:
                    break
        
    return chunks

# -------------------------------------------------------------------

def create_embedding(text):
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = text
    )

    return response.data[0].embedding


# --------------------------semantic processing-----------------------


def semantic_chunking(chunks):
    document_embeddings = []
    for chunk in chunks:
        embedding = create_embedding(chunk["text"])
        document_embeddings.append(embedding)


    document_embeddings = np.array(
            document_embeddings
        ).astype("float32")
    return document_embeddings

# -------------------------semantic retrival-----------------------------------------------

def semantic_retrieve_documents(query, top_k, index):

        query_embedding = create_embedding(query)
        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")

        distances, indicies = index.search(query_embedding, top_k)
        retrieved_chunks = []
        for i in range(top_k):
            chunk_index = indicies[0][i]
            retrieved_chunks.append({
                "chunk": chunks[chunk_index],
                "distance": distances[0][i]
            })

        return retrieved_chunks   


# -------------------------tfidf processing-----------------------------

vectorizer = TfidfVectorizer(stop_words="english")

def tfidf_retrieve_documents(query, top_k, document_vectors):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    top_indicies = similarities.argsort()[-top_k:][::-1]
    retrieved_chunks = []

    for index in top_indicies:
        retrieved_chunks.append({
            "chunk": chunks[index],
            "score": similarities[index]
        })

    return retrieved_chunks

# ------------------------------------------------------------------------

if uploaded_file:

    text = extract_text_from_pdf(uploaded_file)
    chunks = create_chunks(text, 300, 50)

    if not chunks:
        st.error("No text could be extracted from the PDF.")
        st.stop()

    st.title("Basic RAG - reranking")
    st.write("Ask questions")
    query = st.text_input("Enter your question: ")
    document_embeddings = semantic_chunking(chunks)       #does semnatic chunking and add to the faiss index
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(document_embeddings)

    if query:
        semantic_retrieved_chunks = semantic_retrieve_documents(query, 10, index)


        chunk_texts = [chunk["text"] for chunk in chunks]
        document_vectors = vectorizer.fit_transform(chunk_texts)
        tfidf_retrieved_chunks = tfidf_retrieve_documents(query, 10, document_vectors)


        # 12. RECIPROCAL RANK FUSION

        def reciprocal_rank_fusion(semantic_results, keyword_results, k):
            rrf_scores = {}
            chunk_data = {}

            # Process Semantic Ranking

            for rank, result in enumerate(semantic_results):
                chunk_id = result["chunk"]["chunk_id"]
                rrf_scores[chunk_id] = ( rrf_scores.get(chunk_id, 0) +  1 / (k + rank + 1)) * 0.4
                chunk_data[chunk_id] = result

            # Process Keyword Ranking

            for rank, result in enumerate(keyword_results):
                chunk_id = result["chunk"]["chunk_id"]
                rrf_scores[chunk_id] = (rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)) * 0.6
                chunk_data[chunk_id] = result

            # Sort by RRF Score

            sorted_chunks = sorted(
                rrf_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Create Final Results
            final_results = []

            for chunk_id, score in sorted_chunks:
                result = chunk_data[chunk_id].copy()
                result["rrf_score"] = score
                final_results.append(result)

            return final_results


        def generate_answer(query, retrieved_chunks):
            context = "\n\n".join(
                        result["chunk"]["text"]
                        for result in retrieved_chunks
                        )

            prompt = f"""
            You are an assistant.
            Use the context below to answer the user's question.
            If the answer can be reasonably inferred from the context, answer it.

            Only say "The information is not available in the HR policy" when the context contains
            no relevant information at all.

            Context: {context}
            User question: {query}

            Answer:
            """ 
            response = client.chat.completions.create(
                    model = "gpt-4o-mini",

                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0
            )

            return response.choices[0].message.content

   

        if query:
            final_results = reciprocal_rank_fusion(semantic_retrieved_chunks, 
                                                            tfidf_retrieved_chunks, k = 60)
            answer = generate_answer(query, final_results)
            st.subheader("Answer")
            st.write(answer)


            with st.expander("View Retrieved Context"):
                for i, chunk in enumerate(final_results):
                    st.write( f"### Chunk {i+1}" )
                    st.write(chunk["chunk"]["text"])        
        
