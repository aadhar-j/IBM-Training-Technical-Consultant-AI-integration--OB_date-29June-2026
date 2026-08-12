from tools.tavily_tool import tavily_tool


def search_agent(state):
    """
    Search the Internet using Tavily.
    """

    question = state["question"]

    results = tavily_tool(question)

    state["tool_output"] = results

    return state