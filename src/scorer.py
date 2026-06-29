from __future__ import annotations

import re
from typing import Any

from .schemas import CandidateProfile, JobPosting, MatchResult
from .tools import normalize_skill

STOPWORDS = {
    "and",
    "for",
    "the",
    "with",
    "from",
    "this",
    "that",
    "will",
    "intern",
    "internship",
    "candidate",
    "experience",
    "responsibilities",
    "requirements",
}

SOFT_SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "学习速度快": ("学习速度快", "学习能力强", "快速学习", "快速上手", "学习能力", "能快速掌握新技术"),
    "主动查阅资料": ("主动查阅资料", "主动检索资料", "主动查资料", "查阅文档", "阅读文档", "能查资料"),
    "问题拆解能力": ("问题拆解能力", "拆解问题", "问题拆解", "任务拆解", "需求拆解", "分析问题", "问题分析"),
    "自驱力强": ("自驱力强", "自驱力", "自我驱动", "主动性强", "主动学习", "积极主动", "主动推进"),
    "沟通协作能力": ("沟通协作能力", "沟通能力", "团队协作", "团队合作", "协作能力", "跨团队协作"),
    "责任心强": ("责任心强", "责任心", "认真负责", "责任感", "抗压能力", "执行力"),
    "能独立解决问题": ("能独立解决问题", "独立解决问题", "独立定位问题", "独立完成", "解决问题能力", "排查问题"),
}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _normalize_set(skills: list[str]) -> set[str]:
    return {normalize_skill(skill) for skill in skills if isinstance(skill, str) and skill.strip()}


def _ordered_overlap(source_items: list[str], candidate_set: set[str]) -> list[str]:
    overlap = []
    seen = set()
    for item in source_items:
        if not isinstance(item, str):
            continue
        normalized = normalize_skill(item)
        if normalized in candidate_set and normalized not in seen:
            overlap.append(item)
            seen.add(normalized)
    return overlap


def _ordered_missing(source_items: list[str], candidate_set: set[str]) -> list[str]:
    missing = []
    seen = set()
    for item in source_items:
        if not isinstance(item, str):
            continue
        normalized = normalize_skill(item)
        if normalized not in candidate_set and normalized not in seen:
            missing.append(item)
            seen.add(normalized)
    return missing


def _candidate_skill_list(profile: dict) -> list[str]:
    skills = [skill for skill in _as_list(profile.get("skills")) if isinstance(skill, str)]
    skills.extend(skill for skill in _as_list(profile.get("soft_skills")) if isinstance(skill, str))
    for project in _as_list(profile.get("projects")):
        if isinstance(project, dict):
            skills.extend(skill for skill in _as_list(project.get("tech_stack")) if isinstance(skill, str))
    return skills


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9#+.-]*", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _profile_text(profile: dict) -> str:
    parts = []
    parts.extend(str(item) for item in _as_list(profile.get("education")))
    parts.extend(str(item) for item in _as_list(profile.get("skills")))
    parts.extend(str(item) for item in _as_list(profile.get("soft_skills")))
    parts.extend(str(item) for item in _as_list(profile.get("target_roles")))
    parts.extend(str(item) for item in _as_list(profile.get("internships")))
    for project in _as_list(profile.get("projects")):
        if not isinstance(project, dict):
            continue
        parts.append(str(project.get("name") or ""))
        parts.append(str(project.get("description") or ""))
        parts.extend(str(item) for item in _as_list(project.get("tech_stack")))
        parts.extend(str(item) for item in _as_list(project.get("highlights")))
    preferences = profile.get("preferences")
    if isinstance(preferences, dict):
        parts.extend(str(value) for value in preferences.values() if value)
    return " ".join(parts)


def _profile_evidence_text(profile: dict) -> str:
    """Build a broad evidence text from skills, projects, internships, and preferences."""
    return _profile_text(profile).lower()


def _job_text(job: dict) -> str:
    parts = [
        str(job.get("title") or ""),
        str(job.get("company") or ""),
        str(job.get("location") or ""),
        str(job.get("raw_text") or ""),
    ]
    parts.extend(str(item) for item in _as_list(job.get("responsibilities")))
    parts.extend(str(item) for item in _as_list(job.get("required_skills")))
    parts.extend(str(item) for item in _as_list(job.get("preferred_skills")))
    return " ".join(parts)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalized_contains(haystack: str, needle: str) -> bool:
    normalized_haystack = normalize_skill(haystack)
    normalized_needle = normalize_skill(needle)
    return bool(
        normalized_haystack
        and normalized_needle
        and (
            normalized_haystack == normalized_needle
            or normalized_haystack in normalized_needle
            or normalized_needle in normalized_haystack
        )
    )


def _profile_education_text(profile: dict) -> str:
    parts = [str(item) for item in _as_list(profile.get("education"))]
    preferences = profile.get("preferences")
    if isinstance(preferences, dict):
        parts.extend(str(preferences.get(key) or "") for key in ("degree", "major", "education"))
    return " ".join(parts).lower()


def _profile_preference_text(profile: dict) -> str:
    preferences = profile.get("preferences")
    if not isinstance(preferences, dict):
        return ""
    return " ".join(str(value) for value in preferences.values() if value).lower()


def _degree_requirement_satisfied(item_text: str, profile: dict) -> bool:
    education_text = _profile_education_text(profile)
    if not education_text:
        return False

    bachelor_terms = ("本科", "bachelor", "undergraduate")
    master_terms = ("研究生", "硕士", "master", "graduate")
    phd_terms = ("博士", "phd", "doctor")
    high_school_terms = ("高中",)

    has_bachelor_or_above = any(term in education_text for term in bachelor_terms + master_terms + phd_terms)
    has_master_or_above = any(term in education_text for term in master_terms + phd_terms)

    if "本科及以上" in item_text or "本科以上" in item_text:
        return has_bachelor_or_above
    if any(term in item_text for term in master_terms):
        return has_master_or_above
    if any(term in item_text for term in bachelor_terms):
        return has_bachelor_or_above
    if any(term in item_text for term in high_school_terms):
        return True
    return False


def _major_requirement_satisfied(item_text: str, profile: dict) -> bool:
    education_text = _profile_education_text(profile)
    if not education_text:
        return False
    major_terms = (
        "人工智能",
        "计算机",
        "软件工程",
        "自动化",
        "数据科学",
        "数学",
        "电子信息",
    )
    profile_majors = [term for term in major_terms if term in education_text]
    if not profile_majors:
        return False
    if "专业" not in item_text and not any(term in item_text for term in major_terms):
        return False
    return any(term in item_text for term in profile_majors)


def _availability_requirement_satisfied(item_text: str, profile: dict) -> bool:
    preferences = profile.get("preferences") if isinstance(profile.get("preferences"), dict) else {}
    profile_days = _number_from_value(preferences.get("days_per_week"))
    profile_months = _number_from_value(preferences.get("duration_months"))
    preference_text = _profile_preference_text(profile)

    day_numbers = [float(item) for item in re.findall(r"(\d+(?:\.\d+)?)\s*(?:天/周|天|日)", item_text)]
    month_numbers = [float(item) for item in re.findall(r"(\d+(?:\.\d+)?)\s*(?:个月|月)", item_text)]

    days_ok = True
    if day_numbers:
        days_ok = profile_days is not None and profile_days >= max(day_numbers)

    months_ok = True
    if month_numbers:
        months_ok = profile_months is not None and profile_months >= max(month_numbers)

    work_mode_ok = True
    if "线下" in item_text or "到岗" in item_text:
        work_mode_ok = "线下" in preference_text

    has_availability_requirement = bool(day_numbers or month_numbers or "线下" in item_text or "到岗" in item_text)
    return has_availability_requirement and days_ok and months_ok and work_mode_ok


def _structured_requirement_satisfied(item_text: str, profile: dict) -> bool:
    checks = []
    if any(term in item_text for term in ("学历", "本科", "研究生", "硕士", "博士", "高中")):
        checks.append(_degree_requirement_satisfied(item_text, profile))
    if "专业" in item_text or any(term in item_text for term in ("人工智能", "计算机", "软件工程", "自动化", "数据科学")):
        checks.append(_major_requirement_satisfied(item_text, profile))
    if any(term in item_text for term in ("每周", "天/周", "实习周期", "个月", "线下", "到岗")):
        checks.append(_availability_requirement_satisfied(item_text, profile))
    return bool(checks) and all(checks)


def _item_matches_profile(item: str, profile: dict) -> bool:
    """Match one JD requirement/add-on/responsibility against candidate evidence."""
    if not isinstance(item, str) or not item.strip():
        return False

    profile_skills = _candidate_skill_list(profile)
    profile_text = _profile_evidence_text(profile)
    item_text = item.lower()

    if _structured_requirement_satisfied(item_text, profile):
        return True

    if any(_normalized_contains(skill, item) for skill in profile_skills):
        return True

    item_tokens = _tokenize(item)
    profile_tokens = _tokenize(profile_text)
    if item_tokens and item_tokens & profile_tokens:
        return True

    candidate_soft_skills = [
        str(skill).strip()
        for skill in _as_list(profile.get("soft_skills"))
        if isinstance(skill, str) and str(skill).strip()
    ]
    for soft_skill in candidate_soft_skills:
        aliases = SOFT_SKILL_ALIASES.get(soft_skill, (soft_skill,))
        if any(alias.lower() in item_text for alias in aliases):
            return True

    chinese_signals = [
        "工具调用",
        "函数调用",
        "工作流",
        "记忆",
        "反思",
        "自修正",
        "多智能体",
        "知识库",
        "向量",
        "检索",
        "接口",
        "调试",
        "测试",
        "文档",
        "开源",
        "论文",
        "竞赛",
        "学习能力",
        "快速学习",
        "主动性",
        "自驱",
        "问题拆解",
        "拆解问题",
        "沟通协作",
        "团队协作",
        "责任心",
        "独立解决问题",
        "查阅资料",
    ]
    return any(signal in item_text and signal in profile_text for signal in chinese_signals)


def _match_items_against_profile(items: list[str], profile: dict) -> tuple[list[str], list[str]]:
    matched = []
    missing = []
    seen = set()
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        key = normalize_skill(item)
        if key in seen:
            continue
        seen.add(key)
        if _item_matches_profile(item, profile):
            matched.append(item)
        else:
            missing.append(item)
    return matched, missing


def _education_score(profile: dict, job: dict) -> float:
    requirement = str(job.get("education_requirement") or "").lower()
    education_text = " ".join(str(item) for item in _as_list(profile.get("education"))).lower()

    if not requirement or "不限" in requirement:
        return 5.0
    if not education_text:
        return 0.0

    bachelor_terms = ["本科", "bachelor", "undergraduate"]
    master_terms = ["硕士", "master", "graduate"]
    phd_terms = ["博士", "phd", "doctor"]

    if any(term in requirement for term in bachelor_terms):
        return 5.0 if any(term in education_text for term in bachelor_terms + master_terms + phd_terms) else 0.0
    if any(term in requirement for term in master_terms):
        return 5.0 if any(term in education_text for term in master_terms + phd_terms) else 0.0
    if any(term in requirement for term in phd_terms):
        return 5.0 if any(term in education_text for term in phd_terms) else 0.0

    requirement_tokens = _tokenize(requirement)
    education_tokens = _tokenize(education_text)
    return 5.0 if requirement_tokens & education_tokens else 2.5


def _availability_or_experience_score(profile: dict, job: dict) -> float:
    requirement = str(job.get("experience_requirement") or "").lower()
    if not requirement:
        return 5.0
    if "不限" in requirement or "无" in requirement:
        return 5.0

    evidence = _profile_evidence_text(profile)
    if not evidence:
        return 0.0

    requirement_tokens = _tokenize(requirement)
    evidence_tokens = _tokenize(evidence)
    if requirement_tokens & evidence_tokens:
        return 5.0
    if _as_list(profile.get("internships")):
        return 3.0
    return 0.0


def _role_location_score(profile: dict, job: dict) -> float:
    score = 0.0
    title_tokens = _tokenize(str(job.get("title") or ""))
    target_role_tokens = _tokenize(" ".join(str(item) for item in _as_list(profile.get("target_roles"))))
    target_role = profile.get("target_role")
    if target_role:
        target_role_tokens.update(_tokenize(str(target_role)))
    if title_tokens & target_role_tokens:
        score += 3.0

    preferences = profile.get("preferences")
    if isinstance(preferences, dict):
        preferred_location = str(preferences.get("location") or "").lower()
        job_location = str(job.get("location") or "").lower()
        if preferred_location and job_location and (preferred_location in job_location or job_location in preferred_location):
            score += 2.0

    return round(_clamp(score, 0.0, 5.0), 2)


def compute_skill_score(profile_skills: list[str], required_skills: list[str], preferred_skills: list[str]) -> dict:
    candidate_set = _normalize_set(profile_skills)
    required_items = [skill for skill in required_skills if isinstance(skill, str)]
    preferred_items = [skill for skill in preferred_skills if isinstance(skill, str)]

    required_overlap = _ordered_overlap(required_items, candidate_set)
    preferred_overlap = _ordered_overlap(preferred_items, candidate_set)
    overlap = required_overlap + [skill for skill in preferred_overlap if skill not in required_overlap]
    missing = _ordered_missing(required_items + preferred_items, candidate_set)

    required_score = (len(required_overlap) / len(required_items) * 30.0) if required_items else 0.0
    preferred_score = (len(preferred_overlap) / len(preferred_items) * 10.0) if preferred_items else 0.0
    score = round(_clamp(required_score + preferred_score, 0.0, 40.0), 2)

    return {
        "score": score,
        "overlap": overlap,
        "missing": missing,
    }


def compute_project_score(projects: list[dict], job: dict) -> dict:
    job_skills = _as_list(job.get("required_skills")) + _as_list(job.get("preferred_skills"))
    job_skill_set = _normalize_set(job_skills)
    if not job_skill_set:
        return {"score": 0.0, "matched_projects": [], "reason": "JD 中没有可用于项目匹配的技能关键词。"}

    matched_projects = []
    matched_skill_set = set()
    for project in projects:
        if not isinstance(project, dict):
            continue
        tech_stack = _normalize_set(_as_list(project.get("tech_stack")))
        overlap = tech_stack & job_skill_set
        if overlap:
            matched_projects.append(project.get("name") or "未命名项目")
            matched_skill_set.update(overlap)

    score = round(_clamp(len(matched_skill_set) / len(job_skill_set) * 25.0, 0.0, 25.0), 2)
    if matched_projects:
        reason = f"通过 {len(matched_projects)} 个项目覆盖了 {len(matched_skill_set)} 个 JD 技能点。"
    else:
        reason = "项目技术栈暂未覆盖 JD 技能关键词。"

    return {
        "score": score,
        "matched_projects": matched_projects,
        "reason": reason,
    }


def compute_keyword_score(profile: dict, job: dict) -> dict:
    profile_tokens = _tokenize(_profile_text(profile))
    job_tokens = _tokenize(_job_text(job))
    matched_keywords = sorted(profile_tokens & job_tokens)

    denominator = min(max(len(job_tokens), 1), 20)
    score = round(_clamp(len(matched_keywords) / denominator * 10.0, 0.0, 10.0), 2)
    return {
        "score": score,
        "matched_keywords": matched_keywords,
    }


def _matched_projects_for_job(projects: list[dict], job_items: list[str]) -> list[str]:
    matched_projects = []
    for project in projects:
        if not isinstance(project, dict):
            continue
        project_text_parts = [
            str(project.get("name") or ""),
            str(project.get("description") or ""),
        ]
        project_text_parts.extend(str(item) for item in _as_list(project.get("tech_stack")))
        project_text_parts.extend(str(item) for item in _as_list(project.get("highlights")))
        project_profile = {
            "skills": _as_list(project.get("tech_stack")),
            "projects": [project],
            "internships": [],
            "target_roles": [],
            "preferences": {"project_text": " ".join(project_text_parts)},
        }
        if any(_item_matches_profile(item, project_profile) for item in job_items):
            matched_projects.append(project.get("name") or "未命名项目")
    return matched_projects


def _number_from_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _education_score(profile: dict, job: dict) -> float:
    requirement = str(job.get("education_requirement") or job.get("raw_text") or "").lower()
    education_text = " ".join(str(item) for item in _as_list(profile.get("education"))).lower()
    preferences = profile.get("preferences")
    if isinstance(preferences, dict):
        education_text += " " + " ".join(
            str(preferences.get(key) or "") for key in ("degree", "major", "education")
        ).lower()

    if not requirement or "不限" in requirement or "学历不限" in requirement:
        return 5.0
    if not education_text:
        return 0.0

    bachelor_terms = ["本科", "bachelor", "undergraduate"]
    master_terms = ["研究生", "硕士", "master", "graduate"]
    phd_terms = ["博士", "phd", "doctor"]

    if any(term in requirement for term in bachelor_terms):
        return 5.0 if any(term in education_text for term in bachelor_terms + master_terms + phd_terms) else 0.0
    if any(term in requirement for term in master_terms):
        return 5.0 if any(term in education_text for term in master_terms + phd_terms) else 0.0
    if any(term in requirement for term in phd_terms):
        return 5.0 if any(term in education_text for term in phd_terms) else 0.0

    requirement_tokens = _tokenize(requirement)
    education_tokens = _tokenize(education_text)
    return 5.0 if requirement_tokens & education_tokens else 2.5


def _availability_or_experience_score(profile: dict, job: dict) -> float:
    requirement = str(job.get("experience_requirement") or job.get("raw_text") or "").lower()
    if not requirement or "不限" in requirement or "无要求" in requirement:
        return 5.0

    preferences = profile.get("preferences") if isinstance(profile.get("preferences"), dict) else {}
    profile_days = _number_from_value(preferences.get("days_per_week"))
    profile_months = _number_from_value(preferences.get("duration_months"))
    profile_text = _profile_evidence_text(profile)

    day_numbers = [float(item) for item in re.findall(r"(\d+(?:\.\d+)?)\s*(?:天/周|天|日)", requirement)]
    month_numbers = [float(item) for item in re.findall(r"(\d+(?:\.\d+)?)\s*(?:个月|月)", requirement)]
    required_days = max(day_numbers) if day_numbers else None
    required_months = max(month_numbers) if month_numbers else None

    score = 0.0
    if required_days is None:
        score += 2.0
    elif profile_days is not None and profile_days >= required_days:
        score += 2.0

    if required_months is None:
        score += 2.0
    elif profile_months is not None and profile_months >= required_months:
        score += 2.0

    if "线下" in requirement or "到岗" in requirement or "北京" in requirement:
        if "线下" in profile_text or "北京" in profile_text:
            score += 1.0
    else:
        score += 1.0

    if score > 0:
        return round(_clamp(score, 0.0, 5.0), 2)

    evidence = _profile_evidence_text(profile)
    requirement_tokens = _tokenize(requirement)
    evidence_tokens = _tokenize(evidence)
    if requirement_tokens & evidence_tokens:
        return 5.0
    if _as_list(profile.get("internships")):
        return 3.0
    return 0.0


def _role_location_score(profile: dict, job: dict) -> float:
    score = 0.0
    title_tokens = _tokenize(str(job.get("title") or ""))
    target_role_tokens = _tokenize(" ".join(str(item) for item in _as_list(profile.get("target_roles"))))
    target_role = profile.get("target_role")
    if target_role:
        target_role_tokens.update(_tokenize(str(target_role)))
    if title_tokens & target_role_tokens:
        score += 3.0

    preferences = profile.get("preferences")
    if isinstance(preferences, dict):
        preferred_location = " ".join(
            str(preferences.get(key) or "") for key in ("location", "work_mode", "availability")
        ).lower()
        job_location = str(job.get("location") or "").lower()
        if preferred_location and job_location and (preferred_location in job_location or job_location in preferred_location):
            score += 2.0
        elif "线下" in preferred_location and job_location:
            score += 1.0

    return round(_clamp(score, 0.0, 5.0), 2)


def compute_requirement_score(profile: dict, job: dict) -> dict:
    """Score JD 任职要求 against candidate evidence, max 70."""
    required_items = [item for item in _as_list(job.get("required_skills")) if isinstance(item, str)]
    matched_requirements, missing_requirements = _match_items_against_profile(required_items, profile)

    technical_score = (len(matched_requirements) / len(required_items) * 55.0) if required_items else 55.0
    education_score = _education_score(profile, job)
    availability_score = _availability_or_experience_score(profile, job)
    role_location_score = _role_location_score(profile, job)
    score = round(_clamp(technical_score + education_score + availability_score + role_location_score, 0.0, 70.0), 2)

    return {
        "score": score,
        "overlap": matched_requirements,
        "missing": missing_requirements,
        "reason": (
            f"任职要求命中 {len(matched_requirements)}/{len(required_items)} 项；"
            f"学历={education_score}/5，经历或时间={availability_score}/5，方向地点={role_location_score}/5。"
        ),
    }


def compute_bonus_score(profile: dict, job: dict) -> dict:
    """Score JD 加分项 / preferred skills, max 20."""
    preferred_items = [item for item in _as_list(job.get("preferred_skills")) if isinstance(item, str)]
    if not preferred_items:
        return {
            "score": 20.0,
            "overlap": [],
            "missing": [],
            "reason": "JD 未提供明确加分项，默认不扣加分项分数。",
        }

    matched_bonus, missing_bonus = _match_items_against_profile(preferred_items, profile)
    score = round(_clamp(len(matched_bonus) / len(preferred_items) * 20.0, 0.0, 20.0), 2)
    return {
        "score": score,
        "overlap": matched_bonus,
        "missing": missing_bonus,
        "reason": f"加分项命中 {len(matched_bonus)}/{len(preferred_items)} 项。",
    }


def compute_responsibility_score(profile: dict, job: dict) -> dict:
    """Score JD 岗位职责 relatedness, max 10. This is a weak signal, not a hard gate."""
    responsibilities = [item for item in _as_list(job.get("responsibilities")) if isinstance(item, str)]
    if not responsibilities:
        return {
            "score": 0.0,
            "overlap": [],
            "missing": [],
            "reason": "JD 未解析出岗位职责，职责相关性不加分。",
        }

    matched_responsibilities, missing_responsibilities = _match_items_against_profile(responsibilities, profile)
    score = round(_clamp(len(matched_responsibilities) / len(responsibilities) * 10.0, 0.0, 10.0), 2)
    return {
        "score": score,
        "overlap": matched_responsibilities,
        "missing": missing_responsibilities,
        "reason": f"岗位职责相关性命中 {len(matched_responsibilities)}/{len(responsibilities)} 项。",
    }


def _compute_experience_score(profile: dict, job: dict) -> dict:
    internships = _as_list(profile.get("internships"))
    if not internships:
        return {"score": 0.0, "reason": "候选人画像中暂无实习或工作经历。"}

    experience_tokens = _tokenize(" ".join(str(item) for item in internships))
    job_tokens = _tokenize(_job_text(job))
    overlap = sorted(experience_tokens & job_tokens)
    base_score = 5.0 if internships else 0.0
    overlap_score = min(10.0, len(overlap) * 2.0)
    return {
        "score": round(_clamp(base_score + overlap_score, 0.0, 15.0), 2),
        "reason": f"经历关键词重合：{', '.join(overlap) if overlap else '无'}。",
    }


def _compute_preference_score(profile: dict, job: dict) -> dict:
    score = 0.0
    reasons = []
    title_tokens = _tokenize(str(job.get("title") or ""))

    target_role_tokens = set()
    target_role_tokens.update(_tokenize(" ".join(str(item) for item in _as_list(profile.get("target_roles")))))
    target_role = profile.get("target_role")
    if target_role:
        target_role_tokens.update(_tokenize(str(target_role)))

    role_overlap = title_tokens & target_role_tokens
    if role_overlap:
        score += min(6.0, len(role_overlap) * 2.0)
        reasons.append(f"岗位标题与目标方向重合：{', '.join(sorted(role_overlap))}")

    preferences = profile.get("preferences")
    if isinstance(preferences, dict):
        preferred_location = preferences.get("location")
        job_location = job.get("location")
        if preferred_location and job_location and str(preferred_location).lower() in str(job_location).lower():
            score += 4.0
            reasons.append("地点偏好匹配")

    return {
        "score": round(_clamp(score, 0.0, 10.0), 2),
        "reason": "；".join(reasons) if reasons else "暂无明确岗位方向或地点偏好匹配。",
    }


def _recommendation(score: float, missing_skills: list[str]) -> str:
    if score >= 80:
        return "匹配度较高，建议优先投递，并针对 JD 调整项目和技能表述。"
    if score >= 60:
        return "具备一定匹配潜力，建议补强关键技能证据后投递。"
    if missing_skills:
        return f"建议先补足或补充证明这些缺失技能：{', '.join(missing_skills[:3])}。"
    return "匹配优先级较低，建议先复核 JD 与个人目标是否一致。"


def compute_rule_based_match(profile: dict, job: dict) -> dict:
    requirement_result = compute_requirement_score(profile, job)
    bonus_result = compute_bonus_score(profile, job)
    responsibility_result = compute_responsibility_score(profile, job)

    total = round(
        _clamp(
            requirement_result["score"]
            + bonus_result["score"]
            + responsibility_result["score"],
            0.0,
            100.0,
        ),
        2,
    )

    matched_items = (
        requirement_result["overlap"]
        + [item for item in bonus_result["overlap"] if item not in requirement_result["overlap"]]
        + [
            item
            for item in responsibility_result["overlap"]
            if item not in requirement_result["overlap"] and item not in bonus_result["overlap"]
        ]
    )
    missing_items = (
        requirement_result["missing"]
        + [item for item in bonus_result["missing"] if item not in requirement_result["missing"]]
    )
    job_items = (
        _as_list(job.get("required_skills"))
        + _as_list(job.get("preferred_skills"))
        + _as_list(job.get("responsibilities"))
    )
    matched_projects = _matched_projects_for_job(_as_list(profile.get("projects")), job_items)

    reason = (
        "规则评分："
        f"任职要求={requirement_result['score']}/70，"
        f"加分项={bonus_result['score']}/20，"
        f"岗位职责={responsibility_result['score']}/10。"
        f"{requirement_result['reason']}"
        f"{bonus_result['reason']}"
        f"{responsibility_result['reason']}"
    )

    return {
        "job_id": job.get("job_id", ""),
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "match_score": total,
        "skill_overlap": matched_items,
        "missing_skills": missing_items,
        "matched_projects": matched_projects,
        "reason": reason,
        "recommendation": _recommendation(total, requirement_result["missing"]),
    }


def _profile_model_to_dict(profile: CandidateProfile) -> dict:
    if hasattr(profile, "model_dump"):
        return profile.model_dump()
    return profile.dict()


def _job_model_to_dict(job: JobPosting) -> dict:
    if hasattr(job, "model_dump"):
        return job.model_dump()
    return job.dict()


def score_job_match(profile: CandidateProfile, jd: JobPosting) -> MatchResult:
    result = compute_rule_based_match(_profile_model_to_dict(profile), _job_model_to_dict(jd))
    if hasattr(MatchResult, "model_validate"):
        return MatchResult.model_validate(result)
    return MatchResult.parse_obj(result)
