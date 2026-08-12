from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an HR handbook query rewriter.

Rewrite the user's question into a
standalone question suitable for retrieval.

Return only the rewritten question.
"""
        ),

        MessagesPlaceholder(
            variable_name="conversation_history"
        ),

        (
            "human",
            "{question}"
        )
    ]
)


def query_agent(state):

    messages = rewrite_prompt.format_messages(
        question=state["question"],
        conversation_history=state["chat_history"][-5:]
    )

    response = llm.invoke(messages)

    state["question"] = response.content

    return state