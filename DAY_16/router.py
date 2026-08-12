def route_after_evaluation(state):

    score = state["evaluation"]["faithfulness"]

    if score >= 0.95:
        return "good"

    return "clarify"