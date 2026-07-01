from __future__ import annotations

import json
import re
from typing import Any

from .llm_client import call_llm_json
from .tools import normalize_skill


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _normalized_skill_set(skills: list[str]) -> set[str]:
    return {normalize_skill(skill) for skill in skills if isinstance(skill, str) and skill.strip()}


def _profile_skills(profile: dict) -> set[str]:
    skills = _normalized_skill_set(_as_list(profile.get("skills")))
    for project in _as_list(profile.get("projects")):
        if isinstance(project, dict):
            skills.update(_normalized_skill_set(_as_list(project.get("tech_stack"))))
    return skills


def _target_role_keywords(profile: dict) -> set[str]:
    text_parts = []
    text_parts.extend(str(role) for role in _as_list(profile.get("target_roles")))
    target_role = profile.get("target_role")
    if target_role:
        text_parts.append(str(target_role))

    keywords = set()
    for text in text_parts:
        keywords.update(re.findall(r"[a-z0-9]+", text.lower()))
    return {keyword for keyword in keywords if len(keyword) > 1}


def _project_summaries(profile: dict) -> list[str]:
    summaries = []
    for project in _as_list(profile.get("projects")):
        if not isinstance(project, dict):
            continue
        parts = [
            str(project.get("name") or ""),
            str(project.get("description") or ""),
            ", ".join(str(item) for item in _as_list(project.get("tech_stack"))),
            "; ".join(str(item) for item in _as_list(project.get("highlights"))),
        ]
        summary = " ".join(part for part in parts if part).strip()
        if summary:
            summaries.append(summary)
    return summaries


def rule_based_rerank(profile: dict, jobs: list[dict]) -> list[dict]:
    """Rerank jobs with transparent local rules."""
    candidate_skills = _profile_skills(profile)
    target_keywords = _target_role_keywords(profile)
    reranked = []

    for job in jobs:
        if not isinstance(job, dict):
            continue

        required_skills = _normalized_skill_set(_as_list(job.get("required_skills")))
        preferred_skills = _normalized_skill_set(_as_list(job.get("preferred_skills")))
        required_overlap = sorted(required_skills & candidate_skills)
        preferred_overlap = sorted(preferred_skills & candidate_skills)
        missing_required = sorted(required_skills - candidate_skills)

        title_tokens = set(re.findall(r"[a-z0-9]+", str(job.get("title") or "").lower()))
        title_overlap = sorted(title_tokens & target_keywords)

        required_score = len(required_overlap) * 18
        preferred_score = len(preferred_overlap) * 8
        title_score = len(title_overlap) * 6
        missing_penalty = len(missing_required) * 7
        score = max(0.0, min(100.0, required_score + preferred_score + title_score - missing_penalty))

        reason_parts = [
            f"必需技能命中 {len(required_overlap)} 项",
            f"加分技能命中 {len(preferred_overlap)} 项",
            f"岗位标题关键词命中 {len(title_overlap)} 项",
            f"缺失必需技能 {len(missing_required)} 项",
        ]

        item = dict(job)
        item["rerank_score"] = round(score, 2)
        item["rerank_reason"] = "; ".join(reason_parts)
        reranked.append(item)

    reranked.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
    return reranked


def _build_llm_rerank_messages(profile: dict, jobs: list[dict]) -> list[dict]:
    compact_jobs = []
    for job in jobs:
        compact_jobs.append(
            {
                "job_id": job.get("job_id", ""),
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "required_skills": _as_list(job.get("required_skills")),
                "preferred_skills": _as_list(job.get("preferred_skills")),
                "responsibilities": _as_list(job.get("responsibilities"))[:3],
            }
        )

    payload = {
        "candidate": {
            "skills": _as_list(profile.get("skills")),
            "target_roles": _as_list(profile.get("target_roles")),
            "projects_summary": _project_summaries(profile),
        },
        "jobs": compact_jobs,
    }

    system_prompt = """
You are a job reranking assistant for JobPilot-Agent.
Rerank candidate jobs by fit for the candidate.
Return JSON only with this exact shape:
{
  "reranked_jobs": [
    {
      "job_id": "...",
      "rerank_score": 0,
      "rerank_reason": "..."
    }
  ]
}
Scores must be numbers from 0 to 100. Do not invent skills, projects, or job details.
rerank_reason must be written in Chinese.
""".strip()

    user_prompt = f"""
Input data:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return the reranked jobs JSON object.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _validate_llm_rerank_payload(payload: dict) -> list[dict]:
    reranked_jobs = payload.get("reranked_jobs")
    if not isinstance(reranked_jobs, list):
        raise ValueError('LLM JSON 输出必须包含 "reranked_jobs" 列表。')

    validated = []
    for item in reranked_jobs:
        if not isinstance(item, dict):
            raise ValueError("每个重排岗位都必须是 JSON 对象。")
        job_id = item.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("每个重排岗位都必须包含非空 job_id。")
        score = float(item.get("rerank_score"))
        if score < 0 or score > 100:
            raise ValueError("rerank_score 必须在 0 到 100 之间。")
        reason = item.get("rerank_reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("每个重排岗位都必须包含 rerank_reason。")
        validated.append(
            {
                "job_id": job_id,
                "rerank_score": round(score, 2),
                "rerank_reason": reason,
            }
        )
    return validated


def llm_rerank(profile: dict, jobs: list[dict], top_n: int = 5) -> list[dict]:
    """Use DeepSeek to rerank a compact subset of jobs."""
    if top_n <= 0:
        return []

    candidates = [job for job in jobs if isinstance(job, dict)][:top_n]
    if not candidates:
        return []

    messages = _build_llm_rerank_messages(profile, candidates)
    payload = call_llm_json(messages)
    rerank_items = _validate_llm_rerank_payload(payload)

    jobs_by_id = {job.get("job_id"): job for job in candidates if job.get("job_id")}
    output = []
    seen = set()
    for item in rerank_items:
        job = jobs_by_id.get(item["job_id"])
        if job is None:
            continue
        merged = dict(job)
        merged["rerank_score"] = item["rerank_score"]
        merged["rerank_reason"] = item["rerank_reason"]
        output.append(merged)
        seen.add(item["job_id"])

    for job in candidates:
        job_id = job.get("job_id")
        if job_id not in seen:
            output.append(job)

    output.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
    return output


def rerank_jobs(profile: dict, jobs: list[dict], use_llm: bool = False, llm_top_n: int = 5) -> list[dict]:
    """Rerank jobs locally by default, optionally applying LLM reranking to top results."""
    rule_ranked = rule_based_rerank(profile, jobs)
    if not use_llm:
        return rule_ranked

    top_n = min(max(int(llm_top_n or 5), 1), 5, len(rule_ranked))
    llm_ranked = llm_rerank(profile, rule_ranked, top_n=top_n)
    llm_job_ids = {job.get("job_id") for job in llm_ranked}
    remainder = [job for job in rule_ranked if job.get("job_id") not in llm_job_ids]
    return llm_ranked + remainder


def rerank(query: str, documents: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Backward-compatible simple rerank helper."""
    return documents[:top_k]
