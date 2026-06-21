from __future__ import annotations

import json
from typing import Any


def _to_json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_profile_extraction_prompt(user_text: str) -> list[dict]:
    """Build messages for extracting a CandidateProfile from resume-like text."""
    system_prompt = """
You are an information extraction assistant for JobPilot-Agent.
Extract a CandidateProfile from the user's resume, project experience, or personal introduction.

Rules:
- Return JSON only.
- Do not invent or infer information that is not present in the source text.
- Skills, projects, and internships must come directly from the source text.
- If a field is missing, use null for name, [] for list fields, and {} for preferences.
- The JSON object must match this CandidateProfile schema:
{
  "name": string | null,
  "education": [string],
  "skills": [string],
  "projects": [
    {
      "name": string,
      "description": string,
      "tech_stack": [string],
      "highlights": [string]
    }
  ],
  "internships": [string],
  "target_roles": [string],
  "preferences": {}
}
""".strip()

    user_prompt = f"""
Source text:
{user_text}

Return the extracted CandidateProfile as a JSON object.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_jd_extraction_prompt(jd_text: str) -> list[dict]:
    """Build messages for extracting a JobPosting from raw JD text."""
    system_prompt = """
You are a job description extraction assistant for JobPilot-Agent.
Extract structured JobPosting information from the user's JD text.

Rules:
- Return JSON only.
- Do not invent information that is not present in the JD.
- Put only explicit must-have requirements in required_skills.
- Put only nice-to-have or bonus items in preferred_skills.
- Summarize responsibilities as short verb phrases.
- job_id may be an empty string because it will be filled by the tool layer.
- Use null for missing optional scalar fields and [] for missing list fields.
- The JSON object must match this JobPosting schema:
{
  "job_id": string,
  "title": string,
  "company": string,
  "location": string | null,
  "employment_type": string | null,
  "salary": string | null,
  "responsibilities": [string],
  "required_skills": [string],
  "preferred_skills": [string],
  "education_requirement": string | null,
  "experience_requirement": string | null,
  "source_url": string | null,
  "raw_text": string
}
""".strip()

    user_prompt = f"""
JD text:
{jd_text}

Return the extracted JobPosting as a JSON object.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_match_scoring_prompt(profile: dict, job: dict) -> list[dict]:
    """Build messages for scoring a candidate profile against a job posting."""
    system_prompt = """
You are a job matching analyst for JobPilot-Agent.
Score the CandidateProfile against the JobPosting and return a MatchResult.

Scoring rubric, total 100 points:
- Requirement match: 70 points. Use required_skills, education_requirement,
  experience_requirement, location, schedule, and other explicit requirements as the main gate.
- Bonus item match: 20 points. Use preferred_skills or clearly stated bonus/preferred items.
  Missing bonus items should not be treated as hard blockers.
- Responsibility relatedness: 10 points. Use responsibilities as a weak relevance signal:
  responsibilities describe what the intern will do after joining, so they should support the
  explanation and resume advice but must not outweigh requirements.

Rules:
- Return JSON only.
- Base the score only on the provided CandidateProfile and JobPosting.
- Do not invent projects, internships, skills, companies, or preferences.
- match_score must be a number between 0 and 100.
- skill_overlap should contain matched requirement, bonus, or responsibility evidence from the JD.
- missing_skills should prioritize missing hard requirements, then missing bonus items if useful.
- matched_projects should contain project names from the profile that are relevant to the JD.
- reason and recommendation must be written in Chinese.
- The JSON object must match this MatchResult schema:
{
  "job_id": string,
  "title": string,
  "company": string,
  "match_score": number,
  "skill_overlap": [string],
  "missing_skills": [string],
  "matched_projects": [string],
  "reason": string,
  "recommendation": string
}
""".strip()

    user_prompt = f"""
CandidateProfile:
{_to_json_text(profile)}

JobPosting:
{_to_json_text(job)}

Return the MatchResult as a JSON object.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_gap_analysis_prompt(profile: dict, job: dict, match_result: dict) -> list[dict]:
    """Build messages for analyzing gaps between a candidate and a job."""
    system_prompt = """
You are a skill gap analyst for JobPilot-Agent.
Analyze the gaps between the CandidateProfile, JobPosting, and MatchResult.

Rules:
- Return JSON only.
- Do not invent gaps unrelated to the provided data.
- description and suggestion must be written in Chinese.
- Each gap type must be one of:
  "missing_skill", "weak_project_evidence", "no_quantification", "low_keyword_match", "missing_experience"
- Each severity must be one of: "high", "medium", "low"
- The JSON object must use exactly this top-level shape:
{
  "gaps": [
    {
      "type": "missing_skill | weak_project_evidence | no_quantification | low_keyword_match | missing_experience",
      "severity": "high | medium | low",
      "description": string,
      "suggestion": string
    }
  ]
}
""".strip()

    user_prompt = f"""
CandidateProfile:
{_to_json_text(profile)}

JobPosting:
{_to_json_text(job)}

MatchResult:
{_to_json_text(match_result)}

Return the gap analysis as a JSON object.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_resume_suggestion_prompt(profile: dict, job: dict, gaps: list[dict]) -> list[dict]:
    """Build messages for generating resume optimization suggestions."""
    system_prompt = """
You are a resume optimization assistant for JobPilot-Agent.
Generate resume suggestions based on the CandidateProfile, JobPosting, and gap analysis.

Rules:
- Return JSON only.
- Do not invent experience, project outcomes, metrics, or technologies.
- improved_example must be realistic and based on the provided profile evidence.
- If quantified metrics are missing, suggest where the candidate should add real metrics instead of fabricating numbers.
- section, original_problem, suggestion, and improved_example must be written in Chinese.
- The JSON object must use exactly this top-level shape:
{
  "suggestions": [
    {
      "section": string,
      "original_problem": string,
      "suggestion": string,
      "improved_example": string
    }
  ]
}
""".strip()

    user_prompt = f"""
CandidateProfile:
{_to_json_text(profile)}

JobPosting:
{_to_json_text(job)}

Gaps:
{_to_json_text(gaps)}

Return the resume suggestions as a JSON object.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
