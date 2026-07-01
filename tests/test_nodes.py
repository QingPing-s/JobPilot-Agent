
from src import nodes


def _sample_profile():
    return {
        "name": "Alex",
        "education": ["Computer Science"],
        "skills": ["Python", "RAG"],
        "soft_skills": ["学习能力强"],
        "projects": [
            {
                "name": "Mini RAG Assistant",
                "description": "Local document QA.",
                "tech_stack": ["Python", "RAG"],
                "highlights": ["Built retrieval flow."],
            }
        ],
        "internships": [],
        "target_roles": ["RAG Intern"],
        "preferences": {"location": "remote"},
    }


def _sample_job(job_id, title, company="Example AI"):
    return {
        "job_id": job_id,
        "title": title,
        "company": company,
        "location": None,
        "employment_type": None,
        "salary": None,
        "responsibilities": ["Build AI features"],
        "required_skills": ["Python"],
        "preferred_skills": ["RAG"],
        "education_requirement": None,
        "experience_requirement": None,
        "source_url": None,
        "raw_text": f"{title} JD",
    }


def test_profile_node_extracts_from_text(monkeypatch):
    def fake_call_llm_json(messages):
        assert messages[0]["role"] == "system"
        return {
            "name": "Alex",
            "education": ["Computer Science"],
            "skills": ["Python"],
            "soft_skills": [],
            "projects": [],
            "internships": [],
            "target_roles": ["RAG Intern"],
            "preferences": {},
        }

    monkeypatch.setattr(nodes, "call_llm_json", fake_call_llm_json)

    state = nodes.profile_node({"user_profile_text": "Alex knows Python."})

    assert state["candidate_profile"]["name"] == "Alex"
    assert state["node_statuses"]["profile_node"]["status"] == "success"
    assert state["trace"][-1]["node_status"] == "success"


def test_profile_node_uses_fallback_for_text_when_api_unavailable(monkeypatch):
    def fake_call_llm_json(messages):
        raise AssertionError("LLM should not be called when api_available is false.")

    monkeypatch.setattr(nodes, "call_llm_json", fake_call_llm_json)

    state = nodes.profile_node(
        {
            "user_profile_text": "I build Python RAG and LangGraph projects.",
            "target_role": "AI Agent Intern",
            "api_available": False,
        }
    )

    assert state["candidate_profile"]["skills"] == ["Python", "LangGraph", "RAG"]
    assert state["candidate_profile"]["target_roles"] == ["AI Agent Intern"]
    assert state["node_statuses"]["profile_node"]["status"] == "partial"
    assert state["trace"][-1]["fallback_used"] is True


def test_profile_node_records_error_for_missing_input():
    state = nodes.profile_node({})

    assert "candidate_profile" not in state
    assert state["workflow_status"] == "failed"
    assert state["node_statuses"]["profile_node"]["status"] == "error"
    assert state["trace"][-1]["status"] == "error"


def test_jd_parse_node_continues_after_one_file_fails(tmp_path, monkeypatch):
    (tmp_path / "agent_intern_01.txt").write_text("Agent JD", encoding="utf-8")
    (tmp_path / "rag_intern_02.txt").write_text("RAG JD", encoding="utf-8")

    def fake_call_llm_json(messages):
        user_content = messages[1]["content"]
        if "RAG JD" in user_content:
            raise RuntimeError("bad jd")
        return {
            "job_id": "",
            "title": "AI Agent Intern",
            "company": "Example AI",
            "location": None,
            "employment_type": None,
            "salary": None,
            "responsibilities": ["Build agent workflows"],
            "required_skills": ["Python"],
            "preferred_skills": [],
            "education_requirement": None,
            "experience_requirement": None,
            "source_url": None,
            "raw_text": "",
        }

    monkeypatch.setattr(nodes, "call_llm_json", fake_call_llm_json)

    state = nodes.jd_parse_node({"jd_folder": str(tmp_path)})

    assert len(state["parsed_jobs"]) == 2
    assert state["parsed_jobs"][1]["job_id"] == "job_rag_intern_02"
    assert state["jd_parse_failure_count"] == 1
    assert state["node_statuses"]["jd_parse_node"]["status"] == "partial"
    assert state["trace"][-1]["fallback_count"] == 1


def test_jd_parse_node_uses_cached_parsed_jobs(monkeypatch):
    cached_job = _sample_job("job_cached", "Cached Agent Intern")

    def fail_load_jd_files(folder):
        raise AssertionError("jd_parse_node should not read JD files when parsed cache is provided.")

    def fail_call_llm_json(messages):
        raise AssertionError("jd_parse_node should not call LLM when parsed cache is provided.")

    monkeypatch.setattr(nodes, "load_jd_files", fail_load_jd_files)
    monkeypatch.setattr(nodes, "call_llm_json", fail_call_llm_json)

    state = nodes.jd_parse_node(
        {
            "skip_jd_parse": True,
            "job_source": "sqlite_job_library",
            "parsed_jobs": [cached_job],
        }
    )

    assert state["parsed_jobs"] == [cached_job]
    assert state["node_statuses"]["jd_parse_node"]["status"] == "success"
    assert state["trace"][-1]["input_count"] == 1
    assert state["trace"][-1]["output_count"] == 1


def test_retrieve_node_writes_retrieved_jobs(monkeypatch, tmp_path):
    jobs = [_sample_job("job_agent", "Agent Intern"), _sample_job("job_rag", "RAG Intern")]

    def fake_hybrid_retrieve(query, jobs, top_k, persist_dir):
        assert "AI Agent Intern" in query
        assert "Python" in query
        fake_hybrid_retrieve.last_stats = {
            "query": query,
            "vector_top_k": top_k,
            "keyword_top_k": top_k,
            "vector_result_count": 1,
            "keyword_result_count": 1,
            "merged_count": 1,
            "final_retrieved_count": 1,
            "vector_error": "",
            "keyword_error": "",
        }
        return [jobs[1]]

    monkeypatch.setattr(nodes, "hybrid_retrieve", fake_hybrid_retrieve)

    state = nodes.retrieve_node(
        {
            "candidate_profile": _sample_profile(),
            "target_role": "AI Agent Intern",
            "parsed_jobs": jobs,
            "vector_store_dir": str(tmp_path),
            "retrieval_top_k": 1,
        }
    )

    assert state["retrieved_jobs"] == [jobs[1]]
    assert state["node_statuses"]["retrieve_node"]["status"] == "success"
    assert state["trace"][-1]["vector_top_k"] == 1
    assert state["trace"][-1]["vector_result_count"] == 1
    assert state["trace"][-1]["keyword_result_count"] == 1


def test_retrieve_node_falls_back_to_all_jobs_when_hybrid_retrieval_fails(monkeypatch, tmp_path):
    jobs = [_sample_job("job_agent", "Agent Intern")]

    def fake_hybrid_retrieve(query, jobs, top_k, persist_dir):
        raise RuntimeError("hybrid unavailable")

    monkeypatch.setattr(nodes, "hybrid_retrieve", fake_hybrid_retrieve)

    state = nodes.retrieve_node(
        {
            "candidate_profile": _sample_profile(),
            "parsed_jobs": jobs,
            "vector_store_dir": str(tmp_path),
        }
    )

    assert state["retrieved_jobs"] == jobs
    assert state["node_statuses"]["retrieve_node"]["status"] == "partial"
    assert state["trace"][-1]["fallback_used"] is True


def test_rerank_node_falls_back_when_rerank_fails(monkeypatch):
    jobs = [_sample_job("job_agent", "Agent Intern")]

    def fake_rerank_jobs(profile, jobs_to_rerank, use_llm=False, llm_top_n=5):
        raise RuntimeError("rerank unavailable")

    monkeypatch.setattr(nodes, "rerank_jobs", fake_rerank_jobs)

    state = nodes.rerank_node(
        {
            "candidate_profile": _sample_profile(),
            "retrieved_jobs": jobs,
        }
    )

    assert state["reranked_jobs"] == jobs
    assert state["node_statuses"]["rerank_node"]["status"] == "partial"


def test_match_score_node_sets_partial_when_llm_reason_fails(monkeypatch):
    def fake_call_llm_json(messages):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(nodes, "call_llm_json", fake_call_llm_json)

    state = {
        "candidate_profile": _sample_profile(),
        "retrieved_jobs": [_sample_job("job_rag", "RAG Intern")],
        "use_llm_match_scoring": True,
    }

    result = nodes.match_score_node(state)

    assert result["matched_jobs"][0]["job_id"] == "job_rag"
    assert result["node_statuses"]["match_score_node"]["status"] == "partial"
    assert result["trace"][-1]["fallback_count"] == 1


def test_match_score_node_sets_error_when_no_jobs_are_scored():
    state = {
        "candidate_profile": _sample_profile(),
        "retrieved_jobs": ["invalid job"],
    }

    result = nodes.match_score_node(state)

    assert result["matched_jobs"] == []
    assert result["workflow_status"] == "failed"
    assert result["halt_reason"] == "No jobs were successfully scored."
    assert result["node_statuses"]["match_score_node"]["status"] == "error"


def test_gap_analysis_node_falls_back_per_job_when_llm_fails(monkeypatch):
    def fake_call_llm_json(messages):
        raise RuntimeError("gap llm unavailable")

    monkeypatch.setattr(nodes, "call_llm_json", fake_call_llm_json)

    state = {
        "candidate_profile": _sample_profile(),
        "parsed_jobs": [_sample_job("job_1", "Job One")],
        "matched_jobs": [
            {
                "job_id": "job_1",
                "title": "Job One",
                "company": "Example AI",
                "match_score": 95,
                "missing_skills": ["LangGraph"],
                "matched_projects": [],
            }
        ],
        "use_llm_deep_analysis": True,
        "llm_node_max_retries": 0,
    }

    result = nodes.gap_analysis_node(state)

    assert result["gaps"][0]["job_id"] == "job_1"
    assert result["node_statuses"]["gap_analysis_node"]["status"] == "partial"
    assert result["trace"][-1]["fallback_count"] == 1


def test_resume_suggestion_node_falls_back_per_job_when_llm_fails(monkeypatch):
    def fake_call_llm_json(messages):
        raise RuntimeError("resume llm unavailable")

    monkeypatch.setattr(nodes, "call_llm_json", fake_call_llm_json)

    state = {
        "candidate_profile": _sample_profile(),
        "parsed_jobs": [_sample_job("job_1", "Job One")],
        "gaps": [
            {
                "job_id": "job_1",
                "gaps": [
                    {
                        "type": "missing_skill",
                        "severity": "medium",
                        "description": "Need LangGraph",
                        "suggestion": "Add LangGraph evidence",
                    }
                ],
            }
        ],
        "use_llm_deep_analysis": True,
        "llm_node_max_retries": 0,
    }

    result = nodes.resume_suggestion_node(state)

    assert result["resume_suggestions"][0]["job_id"] == "job_1"
    assert result["node_statuses"]["resume_suggestion_node"]["status"] == "partial"
    assert result["trace"][-1]["fallback_count"] == 1
