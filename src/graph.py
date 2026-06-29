from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .llm_client import get_token_usage, reset_token_usage
from .nodes import (
    finalize_workflow_node,
    gap_analysis_node,
    halt_workflow_node,
    human_review_node,
    jd_parse_node,
    match_score_node,
    profile_node,
    rerank_node,
    resume_suggestion_node,
    retrieve_node,
    skip_deep_analysis_node,
)
from .state import JobPilotState

DEFAULT_CHECKPOINT_PATH = Path(
    os.getenv("JOBPILOT_CHECKPOINT_DB", "data/jobpilot_checkpoints.sqlite")
)
_MEMORY_CHECKPOINTER = MemorySaver()
_SQLITE_CHECKPOINTERS: dict[str, Any] = {}
_SQLITE_CONNECTIONS: dict[str, sqlite3.Connection] = {}


def _route_after_profile(state: dict) -> str:
    if state.get("workflow_status") in {"cancelled", "timed_out", "failed"}:
        return "halt_workflow_node"
    if isinstance(state.get("candidate_profile"), dict):
        return "jd_parse_node"
    state["halt_reason"] = "候选人画像为空，工作流无法继续。"
    return "halt_workflow_node"


def _route_after_jd_parse(state: dict) -> str:
    if state.get("workflow_status") in {"cancelled", "timed_out", "failed"}:
        return "halt_workflow_node"
    parsed_jobs = state.get("parsed_jobs")
    if not isinstance(parsed_jobs, list) or not parsed_jobs:
        state["halt_reason"] = "未获得任何有效岗位，请检查岗位库或手动输入的 JD。"
        return "halt_workflow_node"

    try:
        failure_rate = float(state.get("jd_parse_failure_rate") or 0.0)
        threshold = float(state.get("jd_parse_review_threshold") or 0.5)
    except (TypeError, ValueError):
        failure_rate, threshold = 0.0, 0.5
    if bool(state.get("require_human_review_on_parse_failure")) and failure_rate >= threshold:
        return "human_review_node"
    return "retrieve_node"


def _route_after_match(state: dict) -> str:
    if state.get("workflow_status") in {"cancelled", "timed_out", "failed"}:
        return "halt_workflow_node"
    matched_jobs = state.get("matched_jobs")
    if not isinstance(matched_jobs, list) or not matched_jobs:
        state["halt_reason"] = "没有岗位成功完成匹配评分。"
        return "halt_workflow_node"

    try:
        threshold = float(state.get("min_deep_analysis_score") or 35.0)
    except (TypeError, ValueError):
        threshold = 35.0
    best_score = max(float(job.get("match_score", 0)) for job in matched_jobs)
    return "gap_analysis_node" if best_score >= threshold else "skip_deep_analysis_node"


def _route_after_review(state: dict) -> str:
    if state.get("workflow_status") == "running" and not state.get("review_required"):
        return "retrieve_node"
    return "halt_workflow_node"


def _persistent_checkpointer(path: str | Path) -> tuple[Any, str]:
    checkpoint_path = str(Path(path).resolve())
    if checkpoint_path in _SQLITE_CHECKPOINTERS:
        return _SQLITE_CHECKPOINTERS[checkpoint_path], "sqlite"

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        return _MEMORY_CHECKPOINTER, "memory"

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    saver = SqliteSaver(connection)
    _SQLITE_CONNECTIONS[checkpoint_path] = connection
    _SQLITE_CHECKPOINTERS[checkpoint_path] = saver
    return saver, "sqlite"


def build_graph(checkpointer: Any | None = None):
    """Build and compile the conditional JobPilot workflow."""
    workflow = StateGraph(JobPilotState)

    workflow.add_node("profile_node", profile_node)
    workflow.add_node("jd_parse_node", jd_parse_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("rerank_node", rerank_node)
    workflow.add_node("match_score_node", match_score_node)
    workflow.add_node("gap_analysis_node", gap_analysis_node)
    workflow.add_node("resume_suggestion_node", resume_suggestion_node)
    workflow.add_node("human_review_node", human_review_node)
    workflow.add_node("skip_deep_analysis_node", skip_deep_analysis_node)
    workflow.add_node("halt_workflow_node", halt_workflow_node)
    workflow.add_node("finalize_workflow_node", finalize_workflow_node)

    workflow.add_edge(START, "profile_node")
    workflow.add_conditional_edges(
        "profile_node",
        _route_after_profile,
        {
            "jd_parse_node": "jd_parse_node",
            "halt_workflow_node": "halt_workflow_node",
        },
    )
    workflow.add_conditional_edges(
        "jd_parse_node",
        _route_after_jd_parse,
        {
            "retrieve_node": "retrieve_node",
            "human_review_node": "human_review_node",
            "halt_workflow_node": "halt_workflow_node",
        },
    )
    workflow.add_edge("retrieve_node", "rerank_node")
    workflow.add_edge("rerank_node", "match_score_node")
    workflow.add_conditional_edges(
        "match_score_node",
        _route_after_match,
        {
            "gap_analysis_node": "gap_analysis_node",
            "skip_deep_analysis_node": "skip_deep_analysis_node",
            "halt_workflow_node": "halt_workflow_node",
        },
    )
    workflow.add_edge("gap_analysis_node", "resume_suggestion_node")
    workflow.add_edge("resume_suggestion_node", "finalize_workflow_node")
    workflow.add_edge("skip_deep_analysis_node", "finalize_workflow_node")
    workflow.add_edge("finalize_workflow_node", END)
    workflow.add_conditional_edges(
        "human_review_node",
        _route_after_review,
        {
            "retrieve_node": "retrieve_node",
            "halt_workflow_node": "halt_workflow_node",
        },
    )
    workflow.add_edge("halt_workflow_node", END)

    return workflow.compile(checkpointer=checkpointer)


def run_jobpilot(
    initial_state: dict,
    *,
    thread_id: str | None = None,
    checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
) -> dict:
    """Run JobPilot with a resumable LangGraph thread."""
    reset_token_usage()
    state = dict(initial_state)
    run_thread_id = thread_id or str(state.get("thread_id") or uuid4())
    state["thread_id"] = run_thread_id

    if checkpoint_path is None:
        checkpointer, checkpoint_backend = _MEMORY_CHECKPOINTER, "memory"
    else:
        checkpointer, checkpoint_backend = _persistent_checkpointer(checkpoint_path)
    state["checkpoint_backend"] = checkpoint_backend

    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": run_thread_id}}
    result = graph.invoke(state, config=config)
    if result.get("__interrupt__"):
        result["workflow_status"] = "awaiting_review"
        result["review_required"] = True
    result["token_usage"] = get_token_usage()
    result["thread_id"] = run_thread_id
    result["checkpoint_backend"] = checkpoint_backend
    return result


def resume_jobpilot(
    thread_id: str,
    *,
    approved: bool = True,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> dict:
    """Resume a persisted LangGraph thread after an interrupt or review step."""
    reset_token_usage()
    checkpointer, checkpoint_backend = _persistent_checkpointer(checkpoint_path)
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume={"approved": approved}), config=config)
    if result.get("__interrupt__"):
        result["workflow_status"] = "awaiting_review"
        result["review_required"] = True
    result["token_usage"] = get_token_usage()
    result["thread_id"] = thread_id
    result["checkpoint_backend"] = checkpoint_backend
    return result


def run_graph(initial_state: dict) -> dict:
    """Backward-compatible alias for run_jobpilot."""
    return run_jobpilot(initial_state)
