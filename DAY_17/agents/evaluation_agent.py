from ragas_validation import evaluate_answer

def evaluation_agent(state):

    question = state["question"]

    answer = state["answer"]

    documents = state["documents"]

    scores = evaluate_answer(
        question,
        answer,
        documents,
    )

    state["evaluation"] = scores

    return state