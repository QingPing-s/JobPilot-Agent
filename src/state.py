from __future__ import annotations

from typing import TypedDict


class JobPilotState(TypedDict, total=False):
    """Shared state for the JobPilot-Agent LangGraph workflow."""

    user_query: str
    target_role: str
    user_profile_text: str
    user_profile_path: str
    jd_folder: str
    api_available: bool
    vector_store_dir: str
    retrieval_top_k: int
    candidate_profile: dict
    parsed_jobs: list[dict]
    retrieved_jobs: list[dict]
    reranked_jobs: list[dict]
    use_llm_rerank: bool
    use_llm_match_scoring: bool
    matched_jobs: list[dict]
    gaps: list[dict]
    resume_suggestions: list[dict]
    trace: list[dict]
    final_report: dict


AgentState = JobPilotState
