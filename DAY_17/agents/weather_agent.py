from langchain_openai import ChatOpenAI
from tools.weather_tool import weather_tool

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


def weather_agent(state):
    """
    Extract city from question and
    call the weather tool.
    """

    question = state["question"]

    # -----------------------------
    # Extract City
    # -----------------------------
    prompt = f"""
Extract ONLY the city name.

Question:

{question}

Examples:

What is the weather in Bangalore?
Answer:
Bangalore

Weather in Chennai
Answer:
Chennai

Is it raining in Delhi today?
Answer:
Delhi

Return ONLY the city.
"""

    response = llm.invoke(prompt)

    city = response.content.strip()

    # -----------------------------
    # Call Tool
    # -----------------------------
    weather = weather_tool(city)

    # Store tool result
    state["tool_output"] = weather

    return state