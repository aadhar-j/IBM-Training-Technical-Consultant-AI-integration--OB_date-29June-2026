from langgraph.graph import (
    StateGraph,
    END
)

from state import AgentState

from agents.supervisor import (
    supervisor_agent
)

from agents.query_agent import (
    query_agent
)

from agents.retrieval_agent import (
    retrieval_agent
)

from agents.weather_agent import (
    weather_agent
)

from agents.search_agent import (
    search_agent
)

from agents.answer_agent import (
    answer_agent
)

from agents.evaluation_agent import (
    evaluation_agent
)

from agents.clarification_agent import (
    clarification_agent
)

from router import (
    route_question,
    evaluation_router
)


def build_graph():

    g = StateGraph(AgentState)

    # ----------------------------
    # Nodes
    # ----------------------------

    g.add_node(
        "supervisor",
        supervisor_agent
    )

    g.add_node(
        "query",
        query_agent
    )

    g.add_node(
        "retrieve",
        retrieval_agent
    )

    g.add_node(
        "weather",
        weather_agent
    )

    g.add_node(
        "search",
        search_agent
    )

    g.add_node(
        "answer_agent",
        answer_agent
    )

    g.add_node(
        "evaluate",
        evaluation_agent
    )

    g.add_node(
        "clarify",
        clarification_agent
    )

    # ----------------------------
    # Entry Point
    # ----------------------------

    g.set_entry_point(
        "supervisor"
    )

    # ----------------------------
    # Supervisor Routing
    # ----------------------------

    g.add_conditional_edges(
        "supervisor",
        route_question,
        {
            "rag": "query",
            "weather": "weather",
            "search": "search"
        }
    )

    # ----------------------------
    # RAG Flow
    # ----------------------------

    g.add_edge(
        "query",
        "retrieve"
    )

    g.add_edge(
        "retrieve",
        "answer_agent"
    )

    # ----------------------------
    # Weather Flow
    # ----------------------------

    g.add_edge(
        "weather",
        "answer_agent"
    )

    # ----------------------------
    # Search Flow
    # ----------------------------

    g.add_edge(
        "search",
        "answer_agent"
    )

    # ----------------------------
    # Evaluation
    # ----------------------------

    g.add_edge(
        "answer_agent",
        "evaluate"
    )

    g.add_conditional_edges(
        "evaluate",
        evaluation_router,
        {
            "end": END,
            "clarify": "clarify"
        }
    )

    # ----------------------------
    # Clarification
    # ----------------------------

    g.add_edge(
        "clarify",
        END
    )

    return g.compile()