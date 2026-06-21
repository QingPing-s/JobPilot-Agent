from src import reranker


def _profile():
    return {
        "skills": ["Python", "RAG", "LangGraph"],
        "target_roles": ["AI Agent Intern"],
        "projects": [
            {
                "name": "Agent Demo",
                "description": "Tool calling prototype",
                "tech_stack": ["Python", "LLM"],
                "highlights": ["Added trace logging"],
            }
        ],
    }


def _jobs():
    return [
        {
            "job_id": "job_backend",
            "title": "Backend Intern",
            "company": "Example Backend",
            "required_skills": ["Java", "SQL"],
            "preferred_skills": ["Docker"],
            "responsibilities": ["Build APIs"],
        },
        {
            "job_id": "job_agent",
            "title": "AI Agent Intern",
            "company": "Example AI",
            "required_skills": ["Python", "LangGraph"],
            "preferred_skills": ["RAG"],
            "responsibilities": ["Build tool calling workflows"],
        },
    ]


def test_rule_based_rerank_scores_and_sorts():
    results = reranker.rule_based_rerank(_profile(), _jobs())

    assert [job["job_id"] for job in results] == ["job_agent", "job_backend"]
    assert results[0]["rerank_score"] > results[1]["rerank_score"]
    assert "必需技能命中" in results[0]["rerank_reason"]


def test_llm_rerank_merges_scores(monkeypatch):
    def fake_call_llm_json(messages):
        content = messages[1]["content"]
        assert "responsibilities" in content
        return {
            "reranked_jobs": [
                {
                    "job_id": "job_backend",
                    "rerank_score": 80,
                    "rerank_reason": "Relevant backend API work.",
                },
                {
                    "job_id": "job_agent",
                    "rerank_score": 95,
                    "rerank_reason": "Strong agent skill match.",
                },
            ]
        }

    monkeypatch.setattr(reranker, "call_llm_json", fake_call_llm_json)

    results = reranker.llm_rerank(_profile(), _jobs(), top_n=2)

    assert [job["job_id"] for job in results] == ["job_agent", "job_backend"]
    assert results[0]["rerank_score"] == 95.0
    assert results[0]["rerank_reason"] == "Strong agent skill match."


def test_rerank_jobs_uses_rule_based_by_default(monkeypatch):
    def fail_call_llm_json(messages):
        raise AssertionError("LLM should not be called by default")

    monkeypatch.setattr(reranker, "call_llm_json", fail_call_llm_json)

    results = reranker.rerank_jobs(_profile(), _jobs())

    assert results[0]["job_id"] == "job_agent"
    assert "rerank_score" in results[0]


def test_rerank_jobs_optionally_uses_llm(monkeypatch):
    called = []

    def fake_llm_rerank(profile, jobs, top_n=5):
        called.append((profile, jobs, top_n))
        return [dict(jobs[0], rerank_score=99, rerank_reason="LLM top match")]

    monkeypatch.setattr(reranker, "llm_rerank", fake_llm_rerank)

    results = reranker.rerank_jobs(_profile(), _jobs(), use_llm=True)

    assert called
    assert results[0]["rerank_score"] == 99
    assert len(results) == 2
