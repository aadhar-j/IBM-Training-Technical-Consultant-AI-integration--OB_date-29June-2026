import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import faiss
from pypdf import PdfReader

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
                    'page': page_number,
                    'text': chunk_text,
                })
        
                start = end-chunk_overlap       
                if end >= len(text)-1:
                    break
        
    return chunks
    
    
if uploaded_file:

    text = extract_text_from_pdf(uploaded_file)
    chunks = create_chunks(text, 300, 50)

    if not chunks:
        st.error("No text could be extracted from the PDF.")
        st.stop()

    def create_embedding(text):
        response = client.embeddings.create(
            model = "text-embedding-3-small",
            input = text
        )

        return response.data[0].embedding


    document_embeddings = []


    for chunk in chunks:
        embedding = create_embedding(chunk["text"])
        document_embeddings.append(embedding)

    document_embeddings = np.array(
            document_embeddings
        ).astype("float32")

    dimension = document_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(document_embeddings)

    def retrieve_documents(query, top_k=3):

        query_embedding = create_embedding(query)
        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")

        distances, indicies = index.search(query_embedding, top_k)
        retrieved_chunks = []


        for i in range(top_k):
            chunk_index = indicies[0][i]
            retrieved_chunks.append({
                "page": chunks[chunk_index]["page"],
                "text": chunks[chunk_index]["text"],
                "distance": distances[0][i]
            })

        return retrieved_chunks


    def generate_answer(query, retrieved_chunks):
        context = "\n\n".join(
                        f"Page {chunk['page']}:\n{chunk['text']}"
                        for chunk in retrieved_chunks
                    )

        prompt = f"""
        You are an HR policy assistant.
        Use the context below to answer the user's question.
        If the answer can be reasonably inferred from the context, answer it.

        Only say "The information is not available in the document" when the context contains
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
                        "content": "You are a helpful HR policy assistant"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.9
        )

        return response.choices[0].message.content

    query = st.text_input("Enter your question: ")

    if query:
        retrieved_chunks = retrieve_documents(
            query, top_k=3
        )

        answer = generate_answer(
            query, retrieved_chunks
        )

        st.subheader("Answer")
        st.write(answer)


        with st.expander("View Retrieved Context"):
            for chunk in retrieved_chunks:
                st.write(f"Page: {chunk['page']}")
                st.write(chunk["text"])