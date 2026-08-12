def route_question(state):

    return state["route"]


def evaluation_router(state):

    # weather/search never evaluated
    if state["route"] != "rag":
        return "end"

    if not state["evaluation"]:
        return "clarify"

    faithfulness = state["evaluation"].get(
        "faithfulness",
        0
    )

    context_precision = state["evaluation"].get(
        "context_precision",
        0
    )

    if (
        faithfulness >= 0.95
        and context_precision >= 0.95
    ):
        return "end"

    return "clarify"
