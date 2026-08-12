import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)


model = genai.GenerativeModel("gemini-2.5-flash")

st.title("LLM Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Type your question")

if prompt:

    st.session_state.messages.append(
        {"role":"user","content":prompt}
    )

    st.chat_message("user").write(prompt)

    response = model.generate_content(prompt)

    answer = response.text

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )

    st.chat_message("assistant").write(answer)

import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------
# Load Environment Variables
# ---------------------------------------
load_dotenv()
import os

print("OpenRouter Key:", os.getenv("OPENROUTER_API_KEY"))
# ---------------------------------------
# OpenRouter Client
# ---------------------------------------


client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ---------------------------------------
# Available Models
# ---------------------------------------

MODELS = {
    "Google Gemini Flash": "google/gemini-2.5-flash",
    "DeepSeek Chat": "deepseek/deepseek-chat",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
    "Qwen 3": "qwen/qwen3-235b-a22b",
    "Mistral Small": "mistralai/mistral-small"
}

# ---------------------------------------
# Streamlit UI
# ---------------------------------------

st.set_page_config(page_title="Multi Model Chatbot")

st.title(" Multi Model LLM Chatbot")

st.sidebar.header("Settings")

selected_model = st.sidebar.selectbox(
    "Choose LLM",
    list(MODELS.keys())
)

MODEL_NAME = MODELS[selected_model]

st.sidebar.success(f"Using:\n\n{MODEL_NAME}")

# ---------------------------------------
# Chat Memory
# ---------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------
# Display Chat History
# ---------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------------------
# User Input
# ---------------------------------------

prompt = st.chat_input("Ask me anything...")

if prompt:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=st.session_state.messages,

            temperature=0.7,

            max_tokens=1000

        )

        answer = response.choices[0].message.content

    except Exception as e:

        answer = f"Error:\n\n{e}"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.write(answer)