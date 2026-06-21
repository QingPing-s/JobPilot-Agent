from src import graph as graph_module


def test_run_jobpilot_invokes_nodes_in_order(monkeypatch):
    calls = []

    def make_node(name):
        def node(state):
            calls.append(name)
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
    assert len(result["trace"]) == 7


def test_build_graph_returns_compiled_graph():
    compiled = graph_module.build_graph()

    assert hasattr(compiled, "invoke")
