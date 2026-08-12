import os

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

# Load API Key

load_dotenv()

os.environ["SARVAM_API_KEY"] = os.getenv("SARVAM_API_KEY")
SARVAM_BASE_URL = "https://api.sarvam.ai/v1"

# Streamlit

st.set_page_config(page_title="Simple RAG")

st.title("📚 LangChain RAG Demo")

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_pdf:

    with open(uploaded_pdf.name, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    # Load PDF

    loader = PyPDFLoader(uploaded_pdf.name)
    documents = loader.load()
    st.success(f"Loaded {len(documents)} pages")

    # Chunking

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    st.write(f"Chunks created : {len(chunks)}")

    # Embeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Vector Store

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Retriever

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":3}
    )

    # LLM

    llm = ChatOpenAI(
        model="sarvam-105b",
        temperature=0,
        api_key=os.getenv("SARVAM_API_KEY"),
        base_url="https://api.sarvam.ai/v1"
    )

    # RAG Chain

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    question = st.text_input("Ask a question")

    if question:

        with st.spinner("Thinking..."):
            answer = qa.invoke(question)
        st.subheader("Answer")
        st.write(answer["result"])