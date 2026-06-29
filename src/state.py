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
    rerank_top_k: int
    llm_rerank_top_n: int
    llm_match_top_n: int
    gap_top_n: int
    resume_top_n: int
    min_deep_analysis_score: float
    llm_node_max_retries: int
    jd_parse_review_threshold: float
    require_human_review_on_parse_failure: bool
    skip_jd_parse: bool
    job_source: str
    candidate_profile: dict
    parsed_jobs: list[dict]
    retrieved_jobs: list[dict]
    reranked_jobs: list[dict]
    use_llm_rerank: bool
    use_llm_match_scoring: bool
    use_llm_deep_analysis: bool
    deep_analysis: bool
    matched_jobs: list[dict]
    gaps: list[dict]
    resume_suggestions: list[dict]
    trace: list[dict]
    token_usage: dict
    final_report: dict
    workflow_status: str
    halt_reason: str
    review_required: bool
    review_reason: str
    deep_analysis_skipped: bool
    jd_parse_input_count: int
    jd_parse_failure_count: int
    jd_parse_failure_rate: float
    thread_id: str
    checkpoint_backend: str
    run_id: str
    _deadline_epoch: float


AgentState = JobPilotState
