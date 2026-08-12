import os
from tavily import TavilyClient

from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def tavily_tool(question: str):

    response = client.search(

        query=question,

        max_results=5

    )

    results = response["results"]

    text = ""

    for index, item in enumerate(results, start=1):

        text += f"{index}. {item['title']}\n"

        text += f"{item['content']}\n\n"

    return text