import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import faiss

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not loaded. Please check your .env file")
    st.stop()

client = OpenAI(
    api_key=api_key
)


with open('HR_Policy.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chunks = [
    chunk.strip()
    for chunk in text.split("========================================")
    if chunk.strip()
]

# -------------------------------------------------------------------

def create_embedding(text):
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = text
    )

    return response.data[0].embedding

# -------------------------------------------------------------------

document_embeddings = []
for chunk in chunks:
    embedding = create_embedding(chunk)
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
            "text": chunks[chunk_index],
            "distance": distances[0][i]
        })

    return retrieved_chunks

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    prompt = f"""
    You are an HR policy assistant.
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
                    "content": "You are a helpful HR policy assistant"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=1
    )

    return response.choices[0].message.content

st.title("Basic RAG - HR policy assistant")
st.write("Ask questions about HR policy")

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
        for i, chunk in enumerate(
            retrieved_chunks
        ):

            st.write(
                f"### Chunk {i+1}"
            )

            st.write(
                chunk["text"]
            )