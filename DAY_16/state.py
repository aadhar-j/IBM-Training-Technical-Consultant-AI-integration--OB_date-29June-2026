from typing import TypedDict, List, Dict
from langchain_core.documents import Document


class AgentState(TypedDict):

    # User Question
    question: str

    # Retrieved Chunks
    documents: List[Document]

    # Final Answer
    answer: str

    # Evaluation Scores
    evaluation: Dict

    chat_history: list