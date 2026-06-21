import json

from src import nodes


def _sample_profile():
    return {
        "name": "Alex",
        "education": ["Computer Science"],
        "skills": ["Python", "RAG"],
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
            "projects": [],
            "internships": [],
            "target_roles": ["RAG Intern"],
            "preferences": {},
        }

    monkeypatch.setattr(nodes, "call_llm_json", fake_call_llm_json)

    state = nodes.profile_node({"user_profile_text": "Alex knows Python."})

    assert state["candidate_profile"]["name"] == "Alex"
    assert state["trace"][-1]["node"] == "profile_node"
    assert state["trace"][-1]["status"] == "success"


def test_profile_node_loads_from_json_path(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "Alex",
                "education": [],
                "skills": ["Python"],
                "projects": [],
                "internships": [],
                "target_roles": [],
                "preferences": {},
            }
        ),
        encoding="utf-8",
    )

    state = nodes.profile_node({"user_profile_path": str(profile_path)})

    assert state["candidate_profile"]["skills"] == ["Python"]
    assert state["trace"][-1]["status"] == "success"


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
    assert state["trace"][-1]["status"] == "success"


def test_profile_node_records_error_for_missing_input():
    state = nodes.profile_node({})

    assert "candidate_profile" not in state
    trace = state["trace"][-1]
    assert trace["node"] == "profile_node"
    assert trace["event_type"] == "error"
    assert trace["status"] == "error"
    assert trace["input_count"] == 1
    assert trace["output_count"] == 0
    assert trace["message"] == "state 中缺少 user_profile_text 或 user_profile_path。"
    assert trace["error_message"] == "state 中缺少 user_profile_text 或 user_profile_path。"


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

    assert len(state["parsed_jobs"]) == 1
    assert state["parsed_jobs"][0]["job_id"] == "job_agent_intern_01"
    assert state["parsed_jobs"][0]["raw_text"] == "Agent JD"
    assert any(trace["status"] == "error" and "rag_intern_02.txt" in trace["message"] for trace in state["trace"])
    assert state["trace"][-1]["status"] == "success"


def test_match_score_node_sorts_rule_based_results_and_skips_invalid_jobs():
    state = {
        "candidate_profile": _sample_profile(),
        "parsed_jobs": [
            dict(_sample_job("job_backend", "Backend Intern"), required_skills=["Java", "SQL"], preferred_skills=["Docker"]),
            "invalid job",
            _sample_job("job_rag", "RAG Intern"),
        ],
    }

    result = nodes.match_score_node(state)

    assert [job["job_id"] for job in result["matched_jobs"]] == ["job_rag", "job_backend"]
    assert all(0 <= job["match_score"] <= 100 for job in result["matched_jobs"])
    assert any(trace["status"] == "error" and "<无效岗位>" in trace["message"] for trace in result["trace"])
    assert result["trace"][-1]["status"] == "success"


def test_retrieve_node_writes_retrieved_jobs(monkeypatch, tmp_path):
    jobs = [_sample_job("job_agent", "Agent Intern"), _sample_job("job_rag", "RAG Intern")]
    build_calls = []

    def fake_build_chroma_store(parsed_jobs, persist_dir):
        build_calls.append((parsed_jobs, persist_dir))

    def fake_hybrid_retrieve(query, jobs, top_k, persist_dir):
        assert "AI Agent Intern" in query
        assert "Python" in query
        assert len(jobs) == 2
        assert top_k == 1
        fake_hybrid_retrieve.last_stats = {
            "query": query,
            "vector_top_k": top_k,
            "keyword_top_k": top_k,
            "merged_count": 1,
            "final_retrieved_count": 1,
        }
        return [jobs[1]]

    monkeypatch.setattr(nodes, "build_chroma_store", fake_build_chroma_store)
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

    assert build_calls[0][0] == jobs
    assert state["retrieved_jobs"] == [jobs[1]]
    assert state["trace"][-1]["node"] == "retrieve_node"
    assert state["trace"][-1]["status"] == "success"
    assert state["trace"][-1]["vector_top_k"] == 1
    assert state["trace"][-1]["keyword_top_k"] == 1
    assert state["trace"][-1]["merged_count"] == 1
    assert state["trace"][-1]["final_retrieved_count"] == 1


def test_retrieve_node_uses_keyword_when_vector_store_build_fails(monkeypatch, tmp_path):
    jobs = [_sample_job("job_agent", "Agent Intern")]

    def fake_build_chroma_store(parsed_jobs, persist_dir):
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr(nodes, "build_chroma_store", fake_build_chroma_store)

    state = nodes.retrieve_node(
        {
            "candidate_profile": _sample_profile(),
            "parsed_jobs": jobs,
            "vector_store_dir": str(tmp_path),
        }
    )

    assert state["retrieved_jobs"][0]["job_id"] == "job_agent"
    assert state["retrieved_jobs"][0]["retrieve_source"] == "keyword"
    assert state["trace"][-1]["node"] == "retrieve_node"
    assert state["trace"][-1]["status"] == "success"
    assert "向量库构建警告" in state["trace"][-1]["message"]


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
    assert state["trace"][-1]["node"] == "retrieve_node"
    assert state["trace"][-1]["status"] == "error"
    assert "已回退" in state["trace"][-1]["message"]


def test_match_score_node_prefers_retrieved_jobs():
    state = {
        "candidate_profile": _sample_profile(),
        "parsed_jobs": [_sample_job("job_agent", "Agent Intern"), _sample_job("job_rag", "RAG Intern")],
        "retrieved_jobs": [_sample_job("job_rag", "RAG Intern")],
    }

    result = nodes.match_score_node(state)

    assert [job["job_id"] for job in result["matched_jobs"]] == ["job_rag"]
    assert "规则评分" in result["matched_jobs"][0]["reason"]


def test_rerank_node_writes_reranked_jobs(monkeypatch):
    jobs = [_sample_job("job_agent", "Agent Intern"), _sample_job("job_rag", "RAG Intern")]
    calls = []

    def fake_rerank_jobs(profile, jobs_to_rerank, use_llm=False):
        calls.append((profile, jobs_to_rerank, use_llm))
        return [
            dict(jobs_to_rerank[1], rerank_score=91.0, rerank_reason="RAG fit"),
            dict(jobs_to_rerank[0], rerank_score=70.0, rerank_reason="Agent fit"),
        ]

    monkeypatch.setattr(nodes, "rerank_jobs", fake_rerank_jobs)

    state = nodes.rerank_node(
        {
            "candidate_profile": _sample_profile(),
            "target_role": "AI Agent Intern",
            "retrieved_jobs": jobs,
            "use_llm_rerank": True,
        }
    )

    assert [job["job_id"] for job in state["reranked_jobs"]] == ["job_rag", "job_agent"]
    assert calls[0][0]["target_role"] == "AI Agent Intern"
    assert calls[0][2] is True
    assert state["trace"][-1]["node"] == "rerank_node"
    assert state["trace"][-1]["status"] == "success"


def test_match_score_node_prefers_reranked_jobs():
    state = {
        "candidate_profile": _sample_profile(),
        "parsed_jobs": [_sample_job("job_parsed", "Parsed Intern")],
        "retrieved_jobs": [_sample_job("job_retrieved", "Retrieved Intern")],
        "reranked_jobs": [_sample_job("job_reranked", "Reranked Intern")],
    }

    result = nodes.match_score_node(state)

    assert [job["job_id"] for job in result["matched_jobs"]] == ["job_reranked"]


def test_match_score_node_optionally_uses_llm_reason(monkeypatch):
    def fake_call_llm_json(messages):
        return {
            "job_id": "job_rag",
            "title": "RAG Intern",
            "company": "Example AI",
            "match_score": 1.0,
            "skill_overlap": [],
            "missing_skills": [],
            "matched_projects": [],
            "reason": "LLM generated reason.",
            "recommendation": "LLM generated recommendation.",
        }

    monkeypatch.setattr(nodes, "call_llm_json", fake_call_llm_json)

    state = {
        "candidate_profile": _sample_profile(),
        "retrieved_jobs": [_sample_job("job_rag", "RAG Intern")],
        "use_llm_match_scoring": True,
    }

    result = nodes.match_score_node(state)

    assert result["matched_jobs"][0]["reason"] == "LLM generated reason."
    assert result["matched_jobs"][0]["recommendation"] == "LLM generated recommendation."
    assert result["matched_jobs"][0]["match_score"] != 1.0


def test_match_score_node_falls_back_when_llm_reason_fails(monkeypatch):
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
    assert "规则评分" in result["matched_jobs"][0]["reason"]
    assert any("LLM 匹配解释生成失败" in trace["message"] for trace in result["trace"])


def test_gap_analysis_node_uses_top_3_and_skips_failures(monkeypatch):
    seen_prompts = []

    def fake_call_llm_json(messages):
        content = messages[1]["content"]
        seen_prompts.append(content)
        if '"job_id": "job_2"' in content:
            raise RuntimeError("bad gap")
        return {
            "gaps": [
                {
                    "type": "missing_skill",
                    "severity": "medium",
                    "description": "Needs stronger LangGraph evidence.",
                    "suggestion": "Add one LangGraph project bullet.",
                }
            ]
        }

    monkeypatch.setattr(nodes, "call_llm_json", fake_call_llm_json)
    state = {
        "candidate_profile": _sample_profile(),
        "parsed_jobs": [
            _sample_job("job_1", "Job One"),
            _sample_job("job_2", "Job Two"),
            _sample_job("job_3", "Job Three"),
            _sample_job("job_4", "Job Four"),
        ],
        "matched_jobs": [
            {"job_id": "job_1", "title": "Job One", "company": "Example AI", "match_score": 95},
            {"job_id": "job_2", "title": "Job Two", "company": "Example AI", "match_score": 90},
            {"job_id": "job_3", "title": "Job Three", "company": "Example AI", "match_score": 85},
            {"job_id": "job_4", "title": "Job Four", "company": "Example AI", "match_score": 80},
        ],
    }

    result = nodes.gap_analysis_node(state)

    assert [item["job_id"] for item in result["gaps"]] == ["job_1", "job_3"]
    assert result["gaps"][0]["gaps"][0]["type"] == "missing_skill"
    assert len(seen_prompts) == 3
    assert not any('"job_id": "job_4"' in prompt for prompt in seen_prompts)
    assert any(trace["status"] == "error" and "job_2" in trace["message"] for trace in result["trace"])
    assert result["trace"][-1]["status"] == "success"


def test_resume_suggestion_node_uses_top_3_gap_results_and_skips_failures(monkeypatch):
    seen_prompts = []

    def fake_call_llm_json(messages):
        content = messages[1]["content"]
        seen_prompts.append(content)
        if '"job_id": "job_2"' in content:
            raise RuntimeError("bad suggestion")
        return {
            "suggestions": [
                {
                    "section": "Projects",
                    "original_problem": "Project impact is vague.",
                    "suggestion": "Add concrete evidence tied to the JD.",
                    "improved_example": "Built a RAG assistant with documented retrieval tests.",
                }
            ]
        }

    monkeypatch.setattr(nodes, "call_llm_json", fake_call_llm_json)
    state = {
        "candidate_profile": _sample_profile(),
        "parsed_jobs": [
            _sample_job("job_1", "Job One"),
            _sample_job("job_2", "Job Two"),
            _sample_job("job_3", "Job Three"),
            _sample_job("job_4", "Job Four"),
        ],
        "gaps": [
            {"job_id": "job_1", "gaps": [{"type": "missing_skill"}]},
            {"job_id": "job_2", "gaps": [{"type": "low_keyword_match"}]},
            {"job_id": "job_3", "gaps": [{"type": "weak_project_evidence"}]},
            {"job_id": "job_4", "gaps": [{"type": "missing_experience"}]},
        ],
    }

    result = nodes.resume_suggestion_node(state)

    assert [item["job_id"] for item in result["resume_suggestions"]] == ["job_1", "job_3"]
    assert result["resume_suggestions"][0]["suggestions"][0]["section"] == "Projects"
    assert len(seen_prompts) == 3
    assert not any('"job_id": "job_4"' in prompt for prompt in seen_prompts)
    assert any(trace["status"] == "error" and "job_2" in trace["message"] for trace in result["trace"])
    assert result["trace"][-1]["status"] == "success"
