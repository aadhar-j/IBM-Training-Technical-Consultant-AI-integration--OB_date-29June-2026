import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

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


with open('HR_Policy.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chunks = [
    chunk.strip()
    for chunk in text.split("========================================")
    if chunk.strip()
]

# # --------- Keyword based------------------
# vectorizer = TfidfVectorizer(stop_words="english")
# document_vectors = vectorizer.fit_transform(chunks)


# # Semantic
document_vectors = sentence_model.encode(chunks)


def retrieve_documents(query, top_k=5):

    # query_vector = vectorizer.transform([query])
    query_vector = sentence_model.encode([query])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    top_indicies = similarities.argsort()[-top_k:][::-1]
    retrieved_chunks = []

    for index in top_indicies:
        retrieved_chunks.append({
            "text": chunks[index],
            "score": similarities[index]
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

    Only say "The information is not available in the HR policy" when the context contains no relevant information at all.

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

            temperature=0
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

            st.write(
                f"Similarity Score: "
                f"{chunk['score']:.4f}"
            )