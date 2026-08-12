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

def supervisor_agent(state):

    question = state["question"]

    prompt = f"""
Classify the user request.

Categories:

rag
- employee handbook
- HR policy
- leave
- payroll
- benefits

weather
- weather
- temperature
- rain
- forecast

search
- current events
- web search
- internet lookup

Return only:
rag
weather
search

Question:
{question}
"""

    response = llm.invoke(prompt)

    state["route"] = (
        response.content.strip().lower()
    )

    return state