from langgraph.graph import StateGraph, END
from state import AgentState
from agents.query_agent import query_agent
from agents.retrieval_agent import retrieval_agent
from agents.answer_agent import answer_agent
from agents.evaluation_agent import evaluation_agent
from router import route_after_evaluation
from agents.clarification_agent import clarification_agent


def build_graph():
    g=StateGraph(AgentState)
    g.add_node('query',query_agent)
    g.add_node('retrieve',retrieval_agent)
    g.add_node('answer_agent',answer_agent)
    g.add_node('evaluate',evaluation_agent)
    g.add_node("clarify", clarification_agent)


    g.set_entry_point('query')

    g.add_edge('query','retrieve')
    g.add_edge('retrieve','answer_agent')
    g.add_edge('answer_agent','evaluate')
    g.add_conditional_edges("evaluate",
                                        route_after_evaluation,
                                        {
                                            "good": END,
                                            "clarify": "clarify"
                                        })

    g.add_edge("clarify", END)    

    return g.compile()