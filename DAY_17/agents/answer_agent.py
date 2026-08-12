from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

load_dotenv()
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an HR assistant.

Answer ONLY using the supplied context.
If the answer is not found, say so.
"""
        ),

        MessagesPlaceholder(
            variable_name="conversation_history"
        ),

        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)


tool_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant.

Answer the user's question using the provided information.
"""
        ),

        MessagesPlaceholder(
            variable_name="conversation_history"
        ),

        (
            "human",
            """
Information:
{tool_output}

Question:
{question}
"""
        )
    ]
)

def answer_agent(state):

    route = state["route"]

    # ----------------------------------
    # RAG Route
    # ----------------------------------
    if route == "rag":

        context = "\n\n".join(
            doc.page_content
            for doc in state["documents"]
        )

        messages = rag_prompt.format_messages(
            context=context,
            question=state["question"],
            conversation_history=state["chat_history"][-3:]
        )

    # ----------------------------------
    # Weather / Search Routes
    # ----------------------------------
    else:

        messages = tool_prompt.format_messages(
            tool_output=state["tool_output"],
            question=state["question"],
            conversation_history=state["chat_history"][-3:]
        )

    response = llm.invoke(messages)

    state["answer"] = response.content

    return state