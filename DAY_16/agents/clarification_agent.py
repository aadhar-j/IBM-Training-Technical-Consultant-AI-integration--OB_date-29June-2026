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

def clarification_agent(state):

    question = state["question"]

    prompt = f"""
The user's question appears ambiguous
or insufficiently specific.

Generate ONE follow-up question that
would help retrieve a better answer.

Question:
{question}

Return only the follow-up question.
"""

    response = llm.invoke(prompt)

    state["answer"] = response.content

    return state