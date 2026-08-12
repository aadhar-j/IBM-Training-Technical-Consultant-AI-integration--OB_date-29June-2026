import os
from openai import OpenAI
import chromadb
from pypdf import PdfReader
import streamlit as st
from dotenv import load_dotenv


# -------------------------------
# Setup
# -------------------------------
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not loaded. Please check your .env file")
    st.stop()

client = OpenAI(
    api_key=api_key
)

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

# -------------------------------
# Step 1: Load PDF
# -------------------------------
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -------------------------------
# Step 2: Chunk Text
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


if uploaded_file:

    text = load_pdf(uploaded_file)
    chunks = chunk_text(text, 300, 50)

    if not chunks:
        st.error("No text could be extracted from the PDF.")
        st.stop()


    # -------------------------------
    # Step 3: Create Embeddings
    # -------------------------------
    def get_embedding(text):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    # -------------------------------
    # Step 4: Store in ChromaDB
    # -------------------------------
    def create_vector_db(chunks):
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(name="rag_demo")

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[chunk]
            )

        return collection

    # -------------------------------
    # Step 5: Retrieve Relevant Docs
    # -------------------------------
    def retrieve(query, collection, top_k=3):
        query_embedding = get_embedding(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results["documents"][0]

    # -------------------------------
    # Step 6: Generate Answer
    # -------------------------------
    def generate_answer(query, context_docs):
        context = "\n\n".join(context_docs)

        prompt = f"""
    You are a helpful assistant.
    Answer the question based ONLY on the context below.

    Context:
    {context}

    Question:
    {query}
    """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    query = st.text_input("Enter your question: ")
    if query:
        collection = create_vector_db(chunks)
        retrieved_chunks = retrieve(query, collection, top_k=3)
        answer = generate_answer(query, retrieved_chunks)

        st.subheader("Answer")
        st.write(answer)


        with st.expander("View Retrieved Context"):
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.write(f"Chunk {i}")
                st.write(chunk)
                st.divider()