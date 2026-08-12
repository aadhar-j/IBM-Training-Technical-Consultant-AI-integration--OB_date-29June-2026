from typing import TypedDict, List, Dict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):

    question: str

    route: str

    documents: List[Document]

    tool_output: str

    answer: str

    evaluation: Dict

    chat_history: List[BaseMessage]