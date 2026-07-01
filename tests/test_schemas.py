import pytest
from pydantic import ValidationError

from src.schemas import (
    CandidateProfile,
    GapItem,
    JobPosting,
    MatchResult,
    ProjectExperience,
    ResumeSuggestion,
)


def test_project_experience_schema():
    project = ProjectExperience(
        name="Mini RAG Assistant",
        description="Local document QA prototype.",
        tech_stack=["Python", "ChromaDB"],
        highlights=["Built retrieval flow."],
    )

    assert project.name == "Mini RAG Assistant"
    assert project.tech_stack == ["Python", "ChromaDB"]


def test_candidate_profile_schema():
    profile = CandidateProfile(
        name="Alex",
        education=["Computer Science"],
        skills=["Python"],
        soft_skills=["学习速度快", "自驱力强"],
        projects=[
            ProjectExperience(
                name="Agent Demo",
                description="Tool-using agent prototype.",
                tech_stack=["Python"],
                highlights=["Added trace logging."],
            )
        ],
        internships=["AI intern at sample company"],
        target_roles=["AI Agent Intern"],
        preferences={"location": "remote"},
    )

    assert profile.name == "Alex"
    assert profile.skills == ["Python"]
    assert profile.soft_skills == ["学习速度快", "自驱力强"]
    assert profile.preferences["location"] == "remote"


def test_job_posting_schema():
    job = JobPosting(
        job_id="job-001",
        title="RAG Intern",
        company="Example AI",
        raw_text="JD text",
        required_skills=["Python"],
    )

    assert job.title == "RAG Intern"
    assert job.required_skills == ["Python"]
    assert job.preferred_skills == []


def test_match_result_schema():
    result = MatchResult(
        job_id="job-001",
        title="RAG Intern",
        company="Example AI",
        match_score=86.5,
        skill_overlap=["Python", "RAG"],
        missing_skills=["LangGraph"],
        matched_projects=["Mini RAG Assistant"],
        reason="Strong RAG overlap.",
        recommendation="Apply after adding LangGraph evidence.",
    )

    assert result.match_score == 86.5
    assert result.missing_skills == ["LangGraph"]


@pytest.mark.parametrize("score", [-1, 101])
def test_match_score_must_be_between_0_and_100(score):
    with pytest.raises(ValidationError):
        MatchResult(
            job_id="job-001",
            title="RAG Intern",
            company="Example AI",
            match_score=score,
            reason="Invalid score.",
            recommendation="Fix score.",
        )


def test_gap_item_schema():
    gap = GapItem(
        type="missing_skill",
        severity="high",
        description="LangGraph is required but absent.",
        suggestion="Add a small LangGraph project.",
    )

    assert gap.type == "missing_skill"
    assert gap.severity == "high"


def test_resume_suggestion_schema():
    suggestion = ResumeSuggestion(
        section="Projects",
        original_problem="Project impact is vague.",
        suggestion="Add quantified retrieval quality evidence.",
        improved_example="Improved retrieval precision by 15% on a 20-document test set.",
    )

    assert suggestion.section == "Projects"
