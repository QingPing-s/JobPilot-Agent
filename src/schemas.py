from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from typing import Literal
except ImportError:  # Python 3.7 compatibility for the local test runner.
    from typing_extensions import Literal

from pydantic import BaseModel, Field


class ProjectExperience(BaseModel):
    name: str = Field(..., description="Project name.")
    description: str = Field(..., description="Brief description of the project scope and outcome.")
    tech_stack: List[str] = Field(default_factory=list, description="Technologies, frameworks, and tools used.")
    highlights: List[str] = Field(default_factory=list, description="Key achievements or evidence from the project.")


class CandidateProfile(BaseModel):
    name: Optional[str] = Field(default=None, description="Candidate name, if available.")
    education: List[str] = Field(default_factory=list, description="Education background entries.")
    skills: List[str] = Field(default_factory=list, description="Candidate skills and keywords.")
    soft_skills: List[str] = Field(
        default_factory=list,
        description="Soft skills and work traits, such as self-drive, learning ability, communication, and problem decomposition.",
    )
    projects: List[ProjectExperience] = Field(default_factory=list, description="Candidate project experiences.")
    internships: List[str] = Field(default_factory=list, description="Internship or work experience summaries.")
    target_roles: List[str] = Field(default_factory=list, description="Target internship roles.")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Job search preferences and constraints.")


class JobPosting(BaseModel):
    job_id: str = Field(..., description="Stable identifier for the job posting.")
    title: str = Field(..., description="Job title.")
    company: str = Field(..., description="Company name.")
    location: Optional[str] = Field(default=None, description="Job location, if provided.")
    employment_type: Optional[str] = Field(default=None, description="Employment type, such as internship or full-time.")
    salary: Optional[str] = Field(default=None, description="Salary or compensation text, if provided.")
    responsibilities: List[str] = Field(default_factory=list, description="Main job responsibilities.")
    required_skills: List[str] = Field(default_factory=list, description="Required skills extracted from the JD.")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills extracted from the JD.")
    education_requirement: Optional[str] = Field(default=None, description="Education requirement text, if provided.")
    experience_requirement: Optional[str] = Field(default=None, description="Experience requirement text, if provided.")
    source_url: Optional[str] = Field(default=None, description="Original job posting URL, if available.")
    raw_text: str = Field(..., description="Original raw JD text.")


class MatchResult(BaseModel):
    job_id: str = Field(..., description="Matched job identifier.")
    title: str = Field(..., description="Matched job title.")
    company: str = Field(..., description="Matched company name.")
    match_score: float = Field(..., ge=0.0, le=100.0, description="Overall match score from 0 to 100.")
    skill_overlap: List[str] = Field(default_factory=list, description="Skills found in both candidate profile and JD.")
    missing_skills: List[str] = Field(default_factory=list, description="Important JD skills missing from the profile.")
    matched_projects: List[str] = Field(default_factory=list, description="Candidate projects relevant to the JD.")
    reason: str = Field(..., description="Short explanation for the match score.")
    recommendation: str = Field(..., description="Recommended next action for this job application.")


class GapItem(BaseModel):
    type: Literal[
        "missing_skill",
        "weak_project_evidence",
        "no_quantification",
        "low_keyword_match",
        "missing_experience",
    ] = Field(..., description="Type of resume or job-match gap.")
    severity: Literal["high", "medium", "low"] = Field(..., description="Gap severity level.")
    description: str = Field(..., description="Human-readable gap description.")
    suggestion: str = Field(..., description="Actionable suggestion to reduce the gap.")


class ResumeSuggestion(BaseModel):
    section: str = Field(..., description="Resume section to improve.")
    original_problem: str = Field(..., description="Problem found in the current resume content.")
    suggestion: str = Field(..., description="Specific rewrite or improvement suggestion.")
    improved_example: str = Field(..., description="Example improved resume wording.")
