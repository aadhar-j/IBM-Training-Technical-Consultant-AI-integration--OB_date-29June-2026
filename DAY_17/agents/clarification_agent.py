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
The user's question could not be answered
reliably from the handbook.

Ask ONE concise follow-up question.

User Question:
{question}

Return only the clarification question.
"""

    response = llm.invoke(prompt)

    state["answer"] = response.content

    return state