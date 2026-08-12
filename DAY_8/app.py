import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load Environment Variables
load_dotenv()

# Get OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI Client
client = OpenAI(
    api_key=api_key
)

@st.cache_data
def get_openai_models():
    models = client.models.list().data

    # Optional: Keep only LLMs used for chat
    chat_models = sorted(
        model.id
        for model in models
        if model.id.startswith(("gpt", "o"))
    )

    return chat_models

models = get_openai_models()



# ---------------------------------------
# Streamlit UI
# ---------------------------------------

st.set_page_config(
    page_title="OpenAI Chatbot"
)

st.title("🤖 OpenAI Chatbot")

selected_model = st.sidebar.selectbox(
    "Select OpenAI Model",
    models,
    index=models.index("gpt-4o-mini") if "gpt-4o-mini" in models else 0
)
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

prompt = st.chat_input(
    "Ask me anything..."
)


# ---------------------------------------
# Process User Input
# ---------------------------------------

if prompt:

    # ---------------------------------------
    # Save User Message
    # ---------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # ---------------------------------------
    # Display User Message
    # ---------------------------------------

    with st.chat_message("user"):

        st.write(prompt)


    # ---------------------------------------
    # Call OpenAI LLM
    # ---------------------------------------

    try:

        response = client.chat.completions.create(
        model=selected_model,
        messages=st.session_state.messages,
        temperature=0.1,
        max_tokens=100,
        )


        # ---------------------------------------
        # Extract LLM Response
        # ---------------------------------------

        answer = response.choices[0].message.content


    except Exception as e:

        answer = f"Error:\n\n{e}"


    # ---------------------------------------
    # Save Assistant Response
    # ---------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # ---------------------------------------
    # Display Assistant Response
    # ---------------------------------------

    with st.chat_message("assistant"):

        st.write(answer)