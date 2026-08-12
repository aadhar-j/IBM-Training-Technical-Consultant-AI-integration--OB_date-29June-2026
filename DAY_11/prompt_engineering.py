import os
import streamlit as st

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

# =====================================
# Load Environment Variables
# =====================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# =====================================
# Streamlit UI
# =====================================

st.set_page_config(page_title="LangChain RAG Demo")

st.title("📚 LangChain RAG Demo")

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

# =====================================
# Prompt Selection
# =====================================

prompt_type = st.selectbox(
    "Select Prompt Engineering Technique",
    [
        "Zero Shot",
        "One Shot",
        "Few Shot",
        "Role Prompting",
        "Instruction Prompting"
    ]
)

# =====================================
# Prompt Templates
# =====================================

zero_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant.

Answer the question ONLY using the given context.

Context:
{context}

Question:
{question}

Answer:
"""
)

one_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Example

Question:
What is Annual Leave?

Answer:
Employees receive 24 annual leave days.

------------------------------------

Context:
{context}

Question:
{question}

Answer:
"""
)

few_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Example 1

Question:
What is Annual Leave?

Answer:
Employees receive 24 annual leave days.

------------------------------------

Example 2

Question:
Can employees work remotely?

Answer:
Employees may work remotely two days per week.

------------------------------------

Context:
{context}

Question:
{question}

Answer:
"""
)

role_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an experienced HR Manager with over 20 years of experience.

Responsibilities:

- Explain company policies professionally
- Use simple language
- Mention page numbers if available

Context:
{context}

Question:
{question}

Answer:
"""
)

instruction_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Instructions:

1. Use ONLY the provided context.
2. Never use outside knowledge.
3. Never hallucinate.
4. If the answer is unavailable, say:
   "I could not find the answer in the document."
5. Mention page numbers whenever possible.
6. Keep the answer within 150 words.

Context:
{context}

Question:
{question}

Answer:
"""
)

# =====================================
# Select Prompt
# =====================================

if prompt_type == "Zero Shot":
    selected_prompt = zero_prompt

elif prompt_type == "One Shot":
    selected_prompt = one_prompt

elif prompt_type == "Few Shot":
    selected_prompt = few_prompt

elif prompt_type == "Role Prompting":
    selected_prompt = role_prompt

else:
    selected_prompt = instruction_prompt

# =====================================
# Process PDF
# =====================================

if uploaded_pdf:

    pdf_path = uploaded_pdf.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    with st.spinner("Loading PDF..."):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    st.success(f"Loaded {len(documents)} pages")

    # =====================================
    # Chunking
    # =====================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    st.write(f"Chunks Created: {len(chunks)}")

    # =====================================
    # Embeddings
    # =====================================

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    # =====================================
    # Vector Store
    # =====================================

    with st.spinner("Creating Vector Database..."):

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # =====================================
    # LLM
    # =====================================

    llm = ChatOpenAI(
        model="sarvam-105b",
        temperature=0,
        api_key=SARVAM_API_KEY,
        base_url="https://api.sarvam.ai/v1"
    )

    # =====================================
    # Retrieval QA Chain
    # =====================================

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": selected_prompt
        }
    )

    # =====================================
    # Question Input
    # =====================================

    question = st.text_input("Ask a question from the PDF")

    if question:

        with st.spinner("Thinking..."):

            response = qa.invoke(
                {"query": question}
            )

        st.subheader("Answer")
        st.write(response["result"])