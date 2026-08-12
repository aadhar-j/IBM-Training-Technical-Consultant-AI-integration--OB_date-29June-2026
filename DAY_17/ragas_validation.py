from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from datasets import Dataset


def evaluate_answer(question, answer, documents):

    contexts = [doc.page_content for doc in documents]

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [answer],  # Placeholder for demo
        }
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = result.to_pandas().iloc[0].to_dict()

    return scores