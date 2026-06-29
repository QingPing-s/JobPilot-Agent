from src import graph as graph_module


def test_run_jobpilot_invokes_nodes_in_order(monkeypatch):
    calls = []

    def make_node(name):
        def node(state):
            calls.append(name)
            if name == "profile_node":
                state["candidate_profile"] = {"skills": ["Python"]}
            elif name == "jd_parse_node":
                state["parsed_jobs"] = [{"job_id": "job_1"}]
            elif name == "match_score_node":
                state["matched_jobs"] = [{"job_id": "job_1", "match_score": 80}]
            state.setdefault("trace", []).append(
                {
                    "node": name,
                    "status": "success",
                    "message": "ok",
                }
            )
            return state

        return node

    monkeypatch.setattr(graph_module, "profile_node", make_node("profile_node"))
    monkeypatch.setattr(graph_module, "jd_parse_node", make_node("jd_parse_node"))
    monkeypatch.setattr(graph_module, "retrieve_node", make_node("retrieve_node"))
    monkeypatch.setattr(graph_module, "rerank_node", make_node("rerank_node"))
    monkeypatch.setattr(graph_module, "match_score_node", make_node("match_score_node"))
    monkeypatch.setattr(graph_module, "gap_analysis_node", make_node("gap_analysis_node"))
    monkeypatch.setattr(graph_module, "resume_suggestion_node", make_node("resume_suggestion_node"))

    result = graph_module.run_jobpilot({"user_query": "Find RAG internships."})

    assert calls == [
        "profile_node",
        "jd_parse_node",
        "retrieve_node",
        "rerank_node",
        "match_score_node",
        "gap_analysis_node",
        "resume_suggestion_node",
    ]
    assert result["user_query"] == "Find RAG internships."
    assert len(result["trace"]) == 8
    assert result["workflow_status"] == "completed"


def test_build_graph_returns_compiled_graph():
    compiled = graph_module.build_graph()

    assert hasattr(compiled, "invoke")


def test_sqlite_checkpoint_resumes_human_review(tmp_path, monkeypatch):
    def profile(state):
        state["candidate_profile"] = {"skills": ["Python"]}
        return state

    def parse(state):
        state["parsed_jobs"] = [{"job_id": "job_1"}]
        state["jd_parse_input_count"] = 1
        state["jd_parse_failure_rate"] = 1.0
        return state

    def retrieve(state):
        state["retrieved_jobs"] = state["parsed_jobs"]
        return state

    def rerank(state):
        state["reranked_jobs"] = state["retrieved_jobs"]
        return state

    def match(state):
        state["matched_jobs"] = [{"job_id": "job_1", "match_score": 80}]
        return state

    def gap(state):
        state["gaps"] = []
        return state

    def resume_suggestions(state):
        state["resume_suggestions"] = []
        return state

    monkeypatch.setattr(graph_module, "profile_node", profile)
    monkeypatch.setattr(graph_module, "jd_parse_node", parse)
    monkeypatch.setattr(graph_module, "retrieve_node", retrieve)
    monkeypatch.setattr(graph_module, "rerank_node", rerank)
    monkeypatch.setattr(graph_module, "match_score_node", match)
    monkeypatch.setattr(graph_module, "gap_analysis_node", gap)
    monkeypatch.setattr(graph_module, "resume_suggestion_node", resume_suggestions)

    checkpoint = tmp_path / "checkpoints.sqlite"
    interrupted = graph_module.run_jobpilot(
        {
            "require_human_review_on_parse_failure": True,
            "jd_parse_review_threshold": 0.5,
        },
        thread_id="review-thread",
        checkpoint_path=checkpoint,
    )
    assert interrupted["workflow_status"] == "awaiting_review"
    assert interrupted["checkpoint_backend"] == "sqlite"

    resumed = graph_module.resume_jobpilot(
        "review-thread",
        approved=True,
        checkpoint_path=checkpoint,
    )
    assert resumed["workflow_status"] == "completed"
    assert resumed["matched_jobs"][0]["job_id"] == "job_1"
