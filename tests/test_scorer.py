from src.schemas import CandidateProfile, JobPosting, ProjectExperience
from src.scorer import (
    compute_bonus_score,
    compute_keyword_score,
    compute_project_score,
    compute_requirement_score,
    compute_responsibility_score,
    compute_rule_based_match,
    compute_skill_score,
    score_job_match,
)


def test_compute_skill_score_caps_at_40_and_tracks_overlap():
    result = compute_skill_score(
        profile_skills=["Python", "RAG", "LangGraph"],
        required_skills=["Python", "LangGraph"],
        preferred_skills=["RAG", "FastAPI"],
    )

    assert result["score"] == 35.0
    assert result["overlap"] == ["Python", "LangGraph", "RAG"]
    assert result["missing"] == ["FastAPI"]


def test_compute_project_score_matches_project_tech_stack():
    projects = [
        {
            "name": "Agent Demo",
            "description": "Tool calling project",
            "tech_stack": ["Python", "LangGraph"],
            "highlights": ["Added trace logging"],
        },
        {
            "name": "Frontend Demo",
            "description": "UI project",
            "tech_stack": ["React"],
            "highlights": [],
        },
    ]
    job = {
        "required_skills": ["Python", "LangGraph"],
        "preferred_skills": ["RAG"],
    }

    result = compute_project_score(projects, job)

    assert result["score"] == 16.67
    assert result["matched_projects"] == ["Agent Demo"]
    assert "覆盖了 2 个 JD 技能点" in result["reason"]


def test_compute_keyword_score_finds_shared_keywords():
    profile = {
        "skills": ["Python", "RAG"],
        "projects": [
            {
                "name": "DeepSeek Agent",
                "description": "Built tool calling workflow",
                "tech_stack": ["LangGraph"],
                "highlights": [],
            }
        ],
    }
    job = {
        "title": "AI Agent Intern",
        "responsibilities": ["Build LangGraph tool calling workflow"],
        "required_skills": ["Python"],
        "preferred_skills": ["RAG"],
    }

    result = compute_keyword_score(profile, job)

    assert result["score"] > 0
    assert "python" in result["matched_keywords"]
    assert "langgraph" in result["matched_keywords"]


def test_compute_requirement_bonus_and_responsibility_scores_follow_new_rubric():
    profile = {
        "education": ["Computer Science undergraduate candidate"],
        "skills": ["Python", "RAG", "LangGraph"],
        "projects": [
            {
                "name": "Agent Workflow Demo",
                "description": "Built LangGraph RAG workflow with tool calling",
                "tech_stack": ["Python", "RAG", "LangGraph"],
                "highlights": ["Implemented trace logging"],
            }
        ],
        "target_roles": ["AI Agent Intern"],
        "preferences": {"location": "北京"},
    }
    job = {
        "title": "AI Agent Intern",
        "location": "北京",
        "education_requirement": "本科",
        "required_skills": ["Python", "RAG", "Tool Calling", "MCP"],
        "preferred_skills": ["LangGraph", "Open Source Contribution"],
        "responsibilities": ["Build Agent workflow", "Optimize RAG retrieval"],
    }

    requirement = compute_requirement_score(profile, job)
    bonus = compute_bonus_score(profile, job)
    responsibility = compute_responsibility_score(profile, job)

    assert 0 <= requirement["score"] <= 70
    assert 0 <= bonus["score"] <= 20
    assert 0 <= responsibility["score"] <= 10
    assert "Python" in requirement["overlap"]
    assert "MCP" in requirement["missing"]
    assert bonus["overlap"] == ["LangGraph"]
    assert responsibility["score"] > 0


def test_compute_rule_based_match_returns_match_result_shape():
    profile = {
        "name": "Alex",
        "skills": ["Python", "RAG"],
        "projects": [
            {
                "name": "Mini RAG Assistant",
                "description": "Local document QA",
                "tech_stack": ["Python", "RAG"],
                "highlights": ["Built retrieval flow"],
            }
        ],
        "internships": ["AI platform intern working with Python APIs"],
        "target_roles": ["AI Agent Intern"],
        "preferences": {"location": "Remote"},
    }
    job = {
        "job_id": "job_agent",
        "title": "AI Agent Intern",
        "company": "Example AI",
        "location": "Remote",
        "responsibilities": ["Build agent workflows"],
        "required_skills": ["Python"],
        "preferred_skills": ["RAG", "LangGraph"],
        "raw_text": "AI Agent Intern Python RAG",
    }

    result = compute_rule_based_match(profile, job)

    assert result["job_id"] == "job_agent"
    assert 0 <= result["match_score"] <= 100
    assert result["skill_overlap"] == ["Python", "RAG", "Build agent workflows"]
    assert result["missing_skills"] == ["LangGraph"]
    assert result["matched_projects"] == ["Mini RAG Assistant"]
    assert "规则评分" in result["reason"]
    assert "任职要求=" in result["reason"]
    assert "加分项=" in result["reason"]
    assert "岗位职责=" in result["reason"]


def test_score_job_match_compatibility_wrapper():
    profile = CandidateProfile(
        name="Alex",
        skills=["Python", "RAG"],
        projects=[
            ProjectExperience(
                name="Demo",
                description="LLM project.",
                tech_stack=["LLM"],
                highlights=["Built a demo."],
            )
        ],
    )
    jd = JobPosting(
        job_id="job-001",
        title="AI Agent Intern",
        company="Example AI",
        raw_text="JD text",
        required_skills=["Python", "LLM"],
        preferred_skills=["LangGraph"],
    )

    result = score_job_match(profile, jd)

    assert result.job_id == "job-001"
    assert 0 <= result.match_score <= 100
    assert result.skill_overlap == ["Python", "LLM"]
    assert result.missing_skills == ["LangGraph"]
