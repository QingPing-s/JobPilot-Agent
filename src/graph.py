from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    gap_analysis_node,
    jd_parse_node,
    match_score_node,
    profile_node,
    retrieve_node,
    rerank_node,
    resume_suggestion_node,
)
from .state import JobPilotState


def build_graph():
    """Build and compile the JobPilot-Agent workflow."""
    workflow = StateGraph(JobPilotState)

    workflow.add_node("profile_node", profile_node)
    workflow.add_node("jd_parse_node", jd_parse_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("rerank_node", rerank_node)
    workflow.add_node("match_score_node", match_score_node)
    workflow.add_node("gap_analysis_node", gap_analysis_node)
    workflow.add_node("resume_suggestion_node", resume_suggestion_node)

    workflow.add_edge(START, "profile_node")
    workflow.add_edge("profile_node", "jd_parse_node")
    workflow.add_edge("jd_parse_node", "retrieve_node")
    workflow.add_edge("retrieve_node", "rerank_node")
    workflow.add_edge("rerank_node", "match_score_node")
    workflow.add_edge("match_score_node", "gap_analysis_node")
    workflow.add_edge("gap_analysis_node", "resume_suggestion_node")
    workflow.add_edge("resume_suggestion_node", END)

    return workflow.compile()


def run_jobpilot(initial_state: dict) -> dict:
    """Run JobPilot-Agent with an initial state."""
    graph = build_graph()
    return graph.invoke(initial_state)


def run_graph(initial_state: dict) -> dict:
    """Backward-compatible alias for run_jobpilot."""
    return run_jobpilot(initial_state)
