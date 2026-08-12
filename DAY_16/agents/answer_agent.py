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

answer_prompt = ChatPromptTemplate.from_messages(
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


def answer_agent(state):

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    messages = answer_prompt.format_messages(
        context=context,
        question=state["question"],
        conversation_history=state["chat_history"][-3:]
    )

    response = llm.invoke(messages)

    state["answer"] = response.content

    return state