import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from agents.retrieval_agent import set_retriever
from graph import build_graph
from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

# Load API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title("LangGraph RAG")

# creating history session variable
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:

    # PDF processing

    with open(pdf.name, "wb") as f:
        f.write(pdf.getbuffer())

    docs = PyPDFLoader(pdf.name).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    ).split_documents(docs)

    vs = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    set_retriever(
        vs.as_retriever(search_kwargs={"k": 3})
    )

    graph = build_graph()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show previous messages
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Ask a question about the handbook..."
    )

    if user_input:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        result = graph.invoke(
            {
                "question": user_input,
                "documents": [],
                "answer": "",
                "evaluation": {},
                "chat_history": st.session_state.chat_history,
            }
        )

        answer = result["answer"]

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):

            st.markdown(answer)

            if "evaluation" in result:

                scores = result["evaluation"]

                with st.expander("Evaluation"):

                    st.write(
                        f"Faithfulness: {scores['faithfulness']:.3f}"
                    )

                    st.write(
                        f"Answer Relevancy: {scores['answer_relevancy']:.3f}"
                    )

                    st.write(
                        f"Context Precision: {scores['context_precision']:.3f}"
                    )

                    st.write(
                        f"Context Recall: {scores['context_recall']:.3f}"
                    )

        st.session_state.chat_history.extend(
            [
                HumanMessage(content=user_input),
                AIMessage(content=answer)
            ]
        )

        st.session_state.chat_history = (
            st.session_state.chat_history[-10:]
        )

        with st.sidebar:

            st.header("Options")

            if st.button("🗑️ Clear Chat"):

                st.session_state.chat_history = []
                st.session_state.messages = []

                st.rerun()