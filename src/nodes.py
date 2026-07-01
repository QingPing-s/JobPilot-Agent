from __future__ import annotations

import time
from typing import Any, Type

from pydantic import BaseModel

from .llm_client import call_llm_json, get_llm_config, get_token_usage
from .prompts import (
    build_gap_analysis_prompt,
    build_jd_extraction_prompt,
    build_match_scoring_prompt,
    build_profile_extraction_prompt,
    build_resume_suggestion_prompt,
)
from .reranker import rerank_jobs
from .retriever import build_retrieval_query, hybrid_retrieve
from .run_control import deadline_exceeded, is_cancelled, publish_event
from .schemas import CandidateProfile, GapItem, JobPosting, MatchResult, ResumeSuggestion
from .scorer import compute_rule_based_match
from .tools import generate_job_id, load_jd_files, load_user_profile
from .trace_logger import _safe_payload, utc_timestamp


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


def _append_trace(
    state: dict,
    node: str,
    status: str,
    message: str,
    input_count: int = 0,
    output_count: int = 0,
    **extra: Any,
) -> None:
    trace = state.setdefault("trace", [])
    if not isinstance(trace, list):
        trace = []
        state["trace"] = trace

    record = {
        "timestamp": utc_timestamp(),
        "node": node,
        "event_type": "error" if status == "error" else "end",
        "status": status,
        "input_count": input_count,
        "output_count": output_count,
        "message": message,
    }
    if status == "error":
        record["error_message"] = message
    record.update(extra)
    safe_record = _safe_payload(record)
    trace.append(safe_record)
    run_id = state.get("run_id")
    if isinstance(run_id, str) and run_id:
        publish_event(run_id, safe_record)


def _node_statuses(state: dict) -> dict[str, dict[str, Any]]:
    statuses = state.setdefault("node_statuses", {})
    if not isinstance(statuses, dict):
        statuses = {}
        state["node_statuses"] = statuses
    return statuses


def _set_node_status(
    state: dict,
    node_name: str,
    status: str,
    message: str,
    input_count: int = 0,
    output_count: int = 0,
    **extra: Any,
) -> None:
    payload = {
        "timestamp": utc_timestamp(),
        "status": status,
        "message": message,
        "input_count": input_count,
        "output_count": output_count,
    }
    payload.update(extra)
    _node_statuses(state)[node_name] = _safe_payload(payload)


def _node_retry_counts(state: dict) -> dict[str, int]:
    retry_counts = state.setdefault("_llm_retry_counts", {})
    if not isinstance(retry_counts, dict):
        retry_counts = {}
        state["_llm_retry_counts"] = retry_counts
    return retry_counts


def _get_retry_count(state: dict, node_name: str) -> int:
    value = _node_retry_counts(state).get(node_name, 0)
    return value if isinstance(value, int) else 0


def _record_node_result(
    state: dict,
    node_name: str,
    node_status: str,
    message: str,
    input_count: int = 0,
    output_count: int = 0,
    trace_status: str | None = None,
    **extra: Any,
) -> None:
    effective_trace_status = trace_status or ("error" if node_status == "error" else "success")
    payload = {"node_status": node_status, **extra}
    _set_node_status(
        state,
        node_name,
        node_status,
        message,
        input_count=input_count,
        output_count=output_count,
        **payload,
    )
    _append_trace(
        state,
        node_name,
        effective_trace_status,
        message,
        input_count=input_count,
        output_count=output_count,
        **payload,
    )


class WorkflowCancelledError(RuntimeError):
    """Raised when an asynchronous run requests cooperative cancellation."""


class WorkflowTimeoutError(TimeoutError):
    """Raised when an asynchronous run passes its configured deadline."""


def _ensure_execution_allowed(state: dict) -> None:
    cancel_event = state.get("_cancel_event")
    if cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set():
        state["workflow_status"] = "cancelled"
        raise WorkflowCancelledError("运行已被用户取消。")
    run_id = state.get("run_id")
    if isinstance(run_id, str) and run_id and is_cancelled(run_id):
        state["workflow_status"] = "cancelled"
        raise WorkflowCancelledError("运行已被用户取消。")

    deadline = state.get("_deadline_epoch")
    if isinstance(deadline, (int, float)) and time.time() >= float(deadline):
        state["workflow_status"] = "timed_out"
        raise WorkflowTimeoutError("运行超过最大允许时间，已停止后续节点。")
    if isinstance(run_id, str) and run_id and deadline_exceeded(run_id):
        state["workflow_status"] = "timed_out"
        raise WorkflowTimeoutError("运行超过最大允许时间，已停止后续节点。")


def _call_llm_json_with_retry(state: dict, node_name: str, messages: list[dict]) -> dict:
    try:
        max_retries = max(0, min(3, int(state.get("llm_node_max_retries", 1))))
    except (TypeError, ValueError):
        max_retries = 1

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        _ensure_execution_allowed(state)
        try:
            return call_llm_json(messages)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            retry_counts = _node_retry_counts(state)
            retry_counts[node_name] = _get_retry_count(state, node_name) + 1
            _append_trace(
                state,
                node_name,
                "warning",
                f"LLM 调用失败，准备进行第 {attempt + 1} 次重试：{exc}",
                input_count=1,
                output_count=0,
                retry_attempt=attempt + 1,
                retry_limit=max_retries,
            )
            time.sleep(min(1.0, 0.25 * (2**attempt)))

    assert last_error is not None
    raise last_error


def _node_observability(started_at: float, token_usage_before: dict[str, int | float]) -> dict[str, Any]:
    usage_after = get_token_usage()
    return {
        "duration_ms": _duration_ms(started_at),
        "llm_calls": max(0, usage_after.get("calls", 0) - token_usage_before.get("calls", 0)),
        "prompt_tokens": max(
            0,
            usage_after.get("prompt_tokens", 0) - token_usage_before.get("prompt_tokens", 0),
        ),
        "completion_tokens": max(
            0,
            usage_after.get("completion_tokens", 0) - token_usage_before.get("completion_tokens", 0),
        ),
        "total_tokens": max(
            0,
            usage_after.get("total_tokens", 0) - token_usage_before.get("total_tokens", 0),
        ),
        "estimated_cost_usd": round(
            max(
                0.0,
                float(usage_after.get("estimated_cost_usd", 0.0))
                - float(token_usage_before.get("estimated_cost_usd", 0.0)),
            ),
            8,
        ),
        "model_name": get_llm_config().model_name,
    }


def _begin_node(
    state: dict, node_name: str, started_at: float
) -> tuple[bool, dict[str, int | float]]:
    token_usage_before = get_token_usage()
    _node_retry_counts(state)[node_name] = 0
    _set_node_status(state, node_name, "running", f"{node_name} started")
    try:
        _ensure_execution_allowed(state)
    except (WorkflowCancelledError, WorkflowTimeoutError) as exc:
        _record_node_result(
            state,
            node_name,
            "error",
            str(exc),
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return False, token_usage_before
    return True, token_usage_before


def _model_validate(model_cls: Type[BaseModel], data: dict[str, Any]) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _validate_gap_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    gaps = payload.get("gaps")
    if not isinstance(gaps, list):
        raise ValueError('LLM JSON 输出必须包含 "gaps" 列表。')

    validated = []
    for gap in gaps:
        if not isinstance(gap, dict):
            raise ValueError("每个差距项都必须是 JSON 对象。")
        validated.append(_model_to_dict(_model_validate(GapItem, gap)))
    return validated


def _validate_resume_suggestion_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    suggestions = payload.get("suggestions")
    if not isinstance(suggestions, list):
        raise ValueError('LLM JSON 输出必须包含 "suggestions" 列表。')

    validated = []
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            raise ValueError("每条简历建议都必须是 JSON 对象。")
        validated.append(_model_to_dict(_model_validate(ResumeSuggestion, suggestion)))
    return validated


def _job_lookup(parsed_jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {job["job_id"]: job for job in parsed_jobs if isinstance(job, dict) and job.get("job_id")}


_COMMON_SKILLS = [
    "Python",
    "LangGraph",
    "RAG",
    "DeepSeek",
    "OpenAI SDK",
    "LLM",
    "ChromaDB",
    "FastAPI",
    "Pydantic",
    "SQL",
    "Docker",
    "Git",
    "Vector Database",
    "Prompt Engineering",
    "Tool Calling",
    "Agent",
]


def _strip_marker(line: str) -> str:
    return line.strip().lstrip("-*•0123456789.、) ").strip()


def _extract_known_skills(text: str) -> list[str]:
    lowered = text.casefold()
    return [skill for skill in _COMMON_SKILLS if skill.casefold() in lowered]


def _fallback_profile_from_text(user_text: str, target_role: str | None = None) -> dict[str, Any]:
    target_roles = [target_role] if target_role else []
    soft_skills = [
        canonical
        for canonical, aliases in {
            "学习速度快": ("学习速度快", "学习能力强", "快速学习", "快速上手"),
            "主动查阅资料": ("主动查阅资料", "主动查资料", "查阅文档", "阅读文档"),
            "问题拆解能力": ("问题拆解能力", "问题拆解", "拆解问题", "任务拆解"),
            "自驱力强": ("自驱力强", "自驱力", "自我驱动", "主动性强", "积极主动"),
            "沟通协作能力": ("沟通协作能力", "沟通能力", "团队协作", "团队合作"),
            "责任心强": ("责任心强", "责任心", "认真负责", "责任感"),
            "能独立解决问题": ("能独立解决问题", "独立解决问题", "解决问题能力"),
        }.items()
        if any(alias in user_text for alias in aliases)
    ]
    return _model_to_dict(
        _model_validate(
            CandidateProfile,
            {
                "name": None,
                "education": [],
                "skills": _extract_known_skills(user_text),
                "soft_skills": soft_skills,
                "projects": [],
                "internships": [],
                "target_roles": target_roles,
                "preferences": {},
            },
        )
    )


def _fallback_parse_jd(filename: str, raw_text: str, index: int) -> dict[str, Any]:
    lines = [_strip_marker(line) for line in raw_text.splitlines() if _strip_marker(line)]
    title = lines[0] if lines else filename.rsplit(".", 1)[0].replace("_", " ").title()
    if title.casefold().startswith(("title:", "title：", "岗位:", "岗位：")):
        title = title.split(":", 1)[-1].split("：", 1)[-1].strip() or title
    company = "未知公司"
    location = None
    salary = None
    duration = None
    education_requirement = None
    sections: dict[str, list[str]] = {"responsibilities": [], "requirements": [], "preferred": []}
    current_section: str | None = None

    for line in lines:
        lowered = line.casefold()
        if ":" in line or "：" in line:
            value = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            if lowered.startswith(("company:", "company：", "公司:", "公司：")):
                company = value or company
                continue
            if lowered.startswith(("location:", "location：", "地点:", "地点：", "工作地点:", "工作地点：")):
                location = value or None
                continue
            if lowered.startswith(("salary:", "salary：", "薪资:", "薪资：")):
                salary = value or None
                continue
            if lowered.startswith(("duration:", "duration：", "周期:", "周期：", "实习周期:", "实习周期：")):
                duration = value or None
                continue
            if lowered.startswith(("education:", "education：", "学历:", "学历：")):
                education_requirement = value or None
                continue
        if lowered.startswith(("company:", "company：", "公司:", "公司：")):
            company = line.split(":", 1)[-1].split("：", 1)[-1].strip() or company
        elif lowered.startswith(("location:", "location：", "地点:", "地点：")):
            location = line.split(":", 1)[-1].split("：", 1)[-1].strip() or None
        elif lowered.startswith(("responsibilities", "responsibility", "工作职责", "岗位职责")):
            current_section = "responsibilities"
        elif lowered.startswith(("requirements", "requirement", "任职要求", "岗位要求")):
            current_section = "requirements"
        elif lowered.startswith(("preferred", "bonus", "加分", "优先")):
            current_section = "preferred"
        elif current_section:
            sections[current_section].append(line)

    all_skills = _extract_known_skills(raw_text)
    preferred_skills = _extract_known_skills("\n".join(sections["preferred"]))
    required_skills = _extract_known_skills("\n".join(sections["requirements"])) or [
        skill for skill in all_skills if skill not in preferred_skills
    ]

    responsibilities = sections["responsibilities"][:5]
    if not responsibilities:
        responsibilities = [
            line
            for line in lines[1:]
            if not line.casefold().startswith(("company", "location", "公司", "地点"))
        ][:5]

    return _model_to_dict(
        _model_validate(
            JobPosting,
            {
                "job_id": generate_job_id(filename, index),
                "title": title,
                "company": company,
                "location": location,
                "employment_type": None,
                "salary": salary,
                "responsibilities": responsibilities,
                "required_skills": required_skills,
                "preferred_skills": preferred_skills,
                "education_requirement": education_requirement,
                "experience_requirement": duration,
                "source_url": None,
                "raw_text": raw_text,
            },
        )
    )


def _fallback_gaps(match_result: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for skill in _as_list_for_node(match_result.get("missing_skills"))[:5]:
        gaps.append(
            _model_to_dict(
                _model_validate(
                    GapItem,
                    {
                        "type": "missing_skill",
                        "severity": "medium",
                        "description": f"简历中还没有清晰体现 {skill} 的使用证据。",
                        "suggestion": f"补充一个小项目、课程实践或项目 bullet，说明你如何实际使用 {skill}。",
                    },
                )
            )
        )

    if not _as_list_for_node(match_result.get("matched_projects")):
        gaps.append(
            _model_to_dict(
                _model_validate(
                    GapItem,
                    {
                        "type": "weak_project_evidence",
                        "severity": "medium",
                        "description": "当前候选人画像与该 JD 相关的项目证据还不够充分。",
                        "suggestion": "补充一条项目经历，明确项目目标、技术栈和可量化结果。",
                    },
                )
            )
        )
    return gaps


def _fallback_resume_suggestions(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not gaps:
        gaps = [
            {
                "description": "简历 bullet 与 JD 的关键词和职责还可以更贴近。",
                "suggestion": "增加岗位相关关键词，并补充可衡量的项目结果。",
            }
        ]

    suggestions = []
    for gap in gaps[:3]:
        suggestions.append(
            _model_to_dict(
                _model_validate(
                    ResumeSuggestion,
                    {
                        "section": "项目经历 / 技能",
                        "original_problem": gap.get("description", "当前证据不够具体。"),
                        "suggestion": gap.get("suggestion", "将简历 bullet 改写得更贴近 JD。"),
                        "improved_example": "使用 Python 构建 RAG 原型，加入结构化日志、检索效果评测和匹配指标，用于验证岗位相关能力。",
                    },
                )
            )
        )
    return suggestions


def profile_node(state: dict) -> dict:
    """Extract or load CandidateProfile and write it to state."""
    node_name = "profile_node"
    started_at = time.perf_counter()
    can_continue, token_usage_before = _begin_node(state, node_name, started_at)
    if not can_continue:
        return state

    try:
        user_profile_text = state.get("user_profile_text")
        user_profile_path = state.get("user_profile_path")
        fallback_used = False

        if user_profile_text:
            if state.get("api_available", True):
                try:
                    messages = build_profile_extraction_prompt(user_profile_text)
                    profile_data = _call_llm_json_with_retry(state, node_name, messages)
                    source = "llm_profile_extraction"
                except Exception as exc:
                    profile_data = _fallback_profile_from_text(user_profile_text, state.get("target_role"))
                    source = "rule_based_profile_fallback"
                    fallback_used = True
                    _append_trace(
                        state,
                        node_name,
                        "warning",
                        f"候选人画像 LLM 抽取失败，已使用本地规则兜底：{exc}",
                        input_count=1,
                        output_count=1,
                        fallback_used=True,
                    )
            else:
                profile_data = _fallback_profile_from_text(user_profile_text, state.get("target_role"))
                source = "rule_based_profile_fallback"
                fallback_used = True
        elif user_profile_path:
            profile_data = load_user_profile(user_profile_path)
            source = user_profile_path
        else:
            raise ValueError("state 缺少 user_profile_text 或 user_profile_path。")

        profile = _model_validate(CandidateProfile, profile_data)
        state["candidate_profile"] = _model_to_dict(profile)
        node_status = "partial" if fallback_used else "success"
        _record_node_result(
            state,
            node_name,
            node_status,
            f"候选人画像已从 {source} 加载。",
            input_count=1,
            output_count=1,
            fallback_used=fallback_used,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
    except Exception as exc:
        state["workflow_status"] = "failed"
        state["halt_reason"] = str(exc)
        _record_node_result(
            state,
            node_name,
            "error",
            str(exc),
            input_count=1,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )

    return state

def jd_parse_node(state: dict) -> dict:
    """Parse JD text files into JobPosting objects and write them to state."""
    node_name = "jd_parse_node"
    started_at = time.perf_counter()
    can_continue, token_usage_before = _begin_node(state, node_name, started_at)
    if not can_continue:
        return state
    cached_jobs = state.get("parsed_jobs")
    if state.get("skip_jd_parse") and isinstance(cached_jobs, list):
        validated_jobs: list[dict[str, Any]] = []
        invalid_count = 0
        for index, job in enumerate(cached_jobs, start=1):
            try:
                if not isinstance(job, dict):
                    raise ValueError(f"cached job #{index} is not a dict")
                validated_jobs.append(_model_to_dict(_model_validate(JobPosting, job)))
            except Exception as exc:
                invalid_count += 1
                _append_trace(
                    state,
                    node_name,
                    "warning",
                    f"缓存岗位 #{index} 校验失败，已跳过：{exc}",
                    input_count=1,
                    output_count=0,
                )
        state["parsed_jobs"] = validated_jobs
        state["jd_parse_input_count"] = len(cached_jobs)
        state["jd_parse_failure_count"] = invalid_count
        state["jd_parse_failure_rate"] = round(invalid_count / len(cached_jobs), 4) if cached_jobs else 0.0

        if not validated_jobs:
            state["workflow_status"] = "failed"
            state["halt_reason"] = "缓存岗位全部校验失败。"
            _record_node_result(
                state,
                node_name,
                "error",
                state["halt_reason"],
                input_count=len(cached_jobs),
                output_count=0,
                source=state.get("job_source", "cached_parsed_jobs"),
                parse_failure_count=invalid_count,
                parse_failure_rate=state["jd_parse_failure_rate"],
                retry_count=_get_retry_count(state, node_name),
                **_node_observability(started_at, token_usage_before),
            )
            return state

        node_status = "partial" if invalid_count else "success"
        _record_node_result(
            state,
            node_name,
            node_status,
            f"已校验并复用 {len(validated_jobs)} 条岗位库 JD。",
            input_count=len(cached_jobs),
            output_count=len(validated_jobs),
            source=state.get("job_source", "cached_parsed_jobs"),
            parse_failure_count=invalid_count,
            parse_failure_rate=state["jd_parse_failure_rate"],
            fallback_used=invalid_count > 0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    parsed_jobs: list[dict[str, Any]] = []
    state["parsed_jobs"] = parsed_jobs

    try:
        jd_folder = state.get("jd_folder", "data/sample_jds")
        jd_files = load_jd_files(jd_folder)
    except Exception as exc:
        state["workflow_status"] = "failed"
        state["halt_reason"] = str(exc)
        _record_node_result(
            state,
            node_name,
            "error",
            str(exc),
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    parse_failure_count = 0
    fallback_count = 0
    hard_failure_count = 0
    for index, jd_file in enumerate(jd_files, start=1):
        try:
            _ensure_execution_allowed(state)
        except (WorkflowCancelledError, WorkflowTimeoutError) as exc:
            state["halt_reason"] = str(exc)
            break
        filename = jd_file["filename"]
        raw_text = jd_file["raw_text"]

        try:
            if state.get("api_available", True):
                messages = build_jd_extraction_prompt(raw_text)
                job_data = _call_llm_json_with_retry(state, node_name, messages)
                job_data["job_id"] = generate_job_id(filename, index)
                job_data["raw_text"] = raw_text
                job = _model_validate(JobPosting, job_data)
                parsed_jobs.append(_model_to_dict(job))
            else:
                parsed_jobs.append(_fallback_parse_jd(filename, raw_text, index))
                fallback_count += 1
                parse_failure_count += 1
        except Exception as exc:
            parse_failure_count += 1
            try:
                parsed_jobs.append(_fallback_parse_jd(filename, raw_text, index))
                fallback_count += 1
                _append_trace(
                    state,
                    node_name,
                    "warning",
                    f"岗位 {filename} 的 LLM 解析失败，已使用本地规则兜底：{exc}",
                    input_count=1,
                    output_count=1,
                    fallback_used=True,
                )
            except Exception as fallback_exc:
                hard_failure_count += 1
                _append_trace(
                    state,
                    node_name,
                    "error",
                    f"岗位 {filename} 解析失败，已跳过：{fallback_exc}",
                    input_count=1,
                    output_count=0,
                )

    input_count = len(jd_files)
    failure_rate = round(parse_failure_count / input_count, 4) if input_count else 1.0
    state["jd_parse_input_count"] = input_count
    state["jd_parse_failure_count"] = parse_failure_count
    state["jd_parse_failure_rate"] = failure_rate

    if not parsed_jobs:
        state["workflow_status"] = "failed"
        state["halt_reason"] = "没有岗位成功解析。"
        _record_node_result(
            state,
            node_name,
            "error",
            state["halt_reason"],
            input_count=input_count,
            output_count=0,
            parse_failure_count=parse_failure_count,
            parse_failure_rate=failure_rate,
            fallback_count=fallback_count,
            hard_failure_count=hard_failure_count,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    node_status = "success"
    if parse_failure_count > 0 or fallback_count > 0:
        node_status = "partial"

    _record_node_result(
        state,
        node_name,
        node_status,
        f"已解析 {len(parsed_jobs)} / {input_count} 条 JD。",
        input_count=input_count,
        output_count=len(parsed_jobs),
        parse_failure_count=parse_failure_count,
        parse_failure_rate=failure_rate,
        fallback_count=fallback_count,
        hard_failure_count=hard_failure_count,
        retry_count=_get_retry_count(state, node_name),
        **_node_observability(started_at, token_usage_before),
    )
    return state

def retrieve_node(state: dict) -> dict:
    """Retrieve Top-K relevant jobs from the existing retrieval store."""
    node_name = "retrieve_node"
    started_at = time.perf_counter()
    can_continue, token_usage_before = _begin_node(state, node_name, started_at)
    if not can_continue:
        return state

    parsed_jobs = state.get("parsed_jobs")
    candidate_profile = state.get("candidate_profile")
    if not isinstance(parsed_jobs, list):
        state["retrieved_jobs"] = []
        _record_node_result(
            state,
            node_name,
            "error",
            "state 缺少 parsed_jobs，无法执行岗位召回。",
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state
    if not isinstance(candidate_profile, dict):
        state["retrieved_jobs"] = parsed_jobs
        _record_node_result(
            state,
            node_name,
            "partial",
            "candidate_profile 缺失，已退化为按原顺序返回岗位。",
            input_count=len(parsed_jobs),
            output_count=len(parsed_jobs),
            fallback_used=True,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    persist_dir = state.get("vector_store_dir", "data/vector_store")
    top_k = state.get("retrieval_top_k", 20)
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 20

    query = build_retrieval_query(candidate_profile, state.get("target_role"))
    try:
        retrieved_jobs = hybrid_retrieve(query=query, jobs=parsed_jobs, top_k=top_k, persist_dir=persist_dir)
        if not retrieved_jobs and parsed_jobs:
            retrieved_jobs = parsed_jobs[:top_k]

        state["retrieved_jobs"] = retrieved_jobs
        stats = getattr(hybrid_retrieve, "last_stats", {})
        _record_node_result(
            state,
            node_name,
            "success",
            f"混合召回完成：从 {len(parsed_jobs)} 条岗位中召回 {len(retrieved_jobs)} 条，Top-K={top_k}。",
            input_count=len(parsed_jobs),
            output_count=len(retrieved_jobs),
            query=stats.get("query", query),
            vector_top_k=stats.get("vector_top_k", top_k),
            keyword_top_k=stats.get("keyword_top_k", top_k),
            vector_result_count=stats.get("vector_result_count", 0),
            keyword_result_count=stats.get("keyword_result_count", 0),
            merged_count=stats.get("merged_count", len(retrieved_jobs)),
            final_retrieved_count=stats.get("final_retrieved_count", len(retrieved_jobs)),
            vector_error=stats.get("vector_error", ""),
            keyword_error=stats.get("keyword_error", ""),
            fusion=stats.get("fusion", "rrf"),
            rrf_k=stats.get("rrf_k"),
            embedding_model=stats.get("embedding_model"),
            index_version=stats.get("index_version"),
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
    except Exception as exc:
        state["retrieved_jobs"] = parsed_jobs
        _record_node_result(
            state,
            node_name,
            "partial",
            f"混合召回失败，已回退为全量岗位：{exc}",
            input_count=len(parsed_jobs),
            output_count=len(parsed_jobs),
            fallback_used=True,
            error_type=type(exc).__name__,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )

    return state

def rerank_node(state: dict) -> dict:
    """Rerank retrieved jobs before expensive LLM match scoring."""
    node_name = "rerank_node"
    started_at = time.perf_counter()
    can_continue, token_usage_before = _begin_node(state, node_name, started_at)
    if not can_continue:
        return state
    candidate_profile = state.get("candidate_profile")
    retrieved_jobs = state.get("retrieved_jobs")

    if not isinstance(candidate_profile, dict):
        state["reranked_jobs"] = []
        _record_node_result(
            state,
            node_name,
            "error",
            "state 缺少 candidate_profile。",
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state
    if not isinstance(retrieved_jobs, list):
        state["reranked_jobs"] = []
        _record_node_result(
            state,
            node_name,
            "error",
            "state 缺少 retrieved_jobs。",
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    use_llm = bool(state.get("use_llm_rerank", False))
    rerank_top_k = state.get("rerank_top_k", len(retrieved_jobs) or 10)
    try:
        rerank_top_k = max(1, int(rerank_top_k))
    except (TypeError, ValueError):
        rerank_top_k = len(retrieved_jobs) or 10
    try:
        llm_top_n = int(state.get("llm_rerank_top_n", 5))
    except (TypeError, ValueError):
        llm_top_n = 5
    llm_top_n = max(1, min(llm_top_n, 5))

    profile_for_rerank = dict(candidate_profile)
    if state.get("target_role"):
        profile_for_rerank.setdefault("target_role", state["target_role"])

    try:
        reranked_jobs = rerank_jobs(
            profile_for_rerank,
            retrieved_jobs,
            use_llm=use_llm,
            llm_top_n=llm_top_n,
        )
        reranked_jobs = reranked_jobs[:rerank_top_k]
        state["reranked_jobs"] = reranked_jobs
        _record_node_result(
            state,
            node_name,
            "success",
            f"已对 {len(retrieved_jobs)} 条召回岗位完成重排，保留 Top {len(reranked_jobs)}。",
            input_count=len(retrieved_jobs),
            output_count=len(reranked_jobs),
            use_llm=use_llm,
            rerank_top_k=rerank_top_k,
            llm_rerank_top_n=llm_top_n,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
    except Exception as exc:
        state["reranked_jobs"] = retrieved_jobs
        _record_node_result(
            state,
            node_name,
            "partial",
            f"岗位重排失败，已保留原召回顺序：{exc}",
            input_count=len(retrieved_jobs),
            output_count=len(retrieved_jobs),
            use_llm=use_llm,
            rerank_top_k=rerank_top_k,
            llm_rerank_top_n=llm_top_n,
            fallback_used=True,
            error_type=type(exc).__name__,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )

    return state

def _as_list_for_node(value: Any) -> list:
    return value if isinstance(value, list) else []


def match_score_node(state: dict) -> dict:
    """Score reranked jobs when available, then retrieved jobs, then parsed jobs."""
    node_name = "match_score_node"
    started_at = time.perf_counter()
    can_continue, token_usage_before = _begin_node(state, node_name, started_at)
    if not can_continue:
        return state
    matched_jobs: list[dict[str, Any]] = []
    state["matched_jobs"] = matched_jobs

    candidate_profile = state.get("candidate_profile")
    parsed_jobs = state.get("parsed_jobs")
    if "reranked_jobs" in state:
        jobs_to_score = state.get("reranked_jobs")
    elif "retrieved_jobs" in state:
        jobs_to_score = state.get("retrieved_jobs")
    else:
        jobs_to_score = parsed_jobs

    if not isinstance(candidate_profile, dict):
        state["workflow_status"] = "failed"
        state["halt_reason"] = "state 缺少 candidate_profile。"
        _record_node_result(
            state,
            node_name,
            "error",
            state["halt_reason"],
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state
    if not isinstance(jobs_to_score, list):
        state["workflow_status"] = "failed"
        state["halt_reason"] = "state 缺少可评分岗位列表。"
        _record_node_result(
            state,
            node_name,
            "error",
            state["halt_reason"],
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    use_llm = bool(state.get("use_llm_match_scoring", False))
    try:
        llm_match_top_n = int(state.get("llm_match_top_n", 3))
    except (TypeError, ValueError):
        llm_match_top_n = 3
    llm_match_top_n = max(0, min(llm_match_top_n, 3))

    failed_count = 0
    fallback_count = 0

    for index, job in enumerate(jobs_to_score):
        try:
            _ensure_execution_allowed(state)
        except (WorkflowCancelledError, WorkflowTimeoutError) as exc:
            state["halt_reason"] = str(exc)
            break
        try:
            if not isinstance(job, dict):
                raise ValueError("岗位数据必须是 dict。")

            match_data = compute_rule_based_match(candidate_profile, job)
            if use_llm and index < llm_match_top_n:
                try:
                    messages = build_match_scoring_prompt(candidate_profile, job)
                    llm_data = _call_llm_json_with_retry(state, node_name, messages)
                    llm_result = _model_to_dict(_model_validate(MatchResult, llm_data))
                    match_data["reason"] = llm_result["reason"]
                    match_data["recommendation"] = llm_result["recommendation"]
                except Exception as exc:
                    fallback_count += 1
                    job_id = job.get("job_id", "<未知岗位>")
                    _append_trace(
                        state,
                        node_name,
                        "warning",
                        f"岗位 {job_id} 的 LLM 解释失败，已保留规则评分结果：{exc}",
                        input_count=1,
                        output_count=1,
                        fallback_used=True,
                    )

            match_result = _model_validate(MatchResult, match_data)
            matched_jobs.append(_model_to_dict(match_result))
        except Exception as exc:
            failed_count += 1
            job_id = job.get("job_id", "<未知岗位>") if isinstance(job, dict) else "<未知岗位>"
            _append_trace(
                state,
                node_name,
                "error",
                f"岗位 {job_id} 评分失败：{exc}",
                input_count=1,
                output_count=0,
            )
            continue

    matched_jobs.sort(key=lambda item: item["match_score"], reverse=True)

    if not matched_jobs:
        state["workflow_status"] = "failed"
        state["halt_reason"] = "No jobs were successfully scored."
        _record_node_result(
            state,
            node_name,
            "error",
            state["halt_reason"],
            input_count=len(jobs_to_score),
            output_count=0,
            failed_count=failed_count,
            fallback_count=fallback_count,
            use_llm=use_llm,
            llm_match_top_n=llm_match_top_n,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    node_status = "success"
    if failed_count > 0 or fallback_count > 0:
        node_status = "partial"

    _record_node_result(
        state,
        node_name,
        node_status,
        f"已完成 {len(matched_jobs)} / {len(jobs_to_score)} 个岗位评分。",
        input_count=len(jobs_to_score),
        output_count=len(matched_jobs),
        failed_count=failed_count,
        fallback_count=fallback_count,
        use_llm=use_llm,
        llm_match_top_n=llm_match_top_n,
        retry_count=_get_retry_count(state, node_name),
        **_node_observability(started_at, token_usage_before),
    )
    return state

def gap_analysis_node(state: dict) -> dict:
    """Generate gap analysis for the top matched jobs."""
    node_name = "gap_analysis_node"
    started_at = time.perf_counter()
    can_continue, token_usage_before = _begin_node(state, node_name, started_at)
    if not can_continue:
        return state
    gap_results: list[dict[str, Any]] = []
    state["gaps"] = gap_results

    candidate_profile = state.get("candidate_profile")
    parsed_jobs = state.get("parsed_jobs")
    matched_jobs = state.get("matched_jobs")

    if not isinstance(candidate_profile, dict):
        _record_node_result(
            state,
            node_name,
            "error",
            "state 缺少 candidate_profile。",
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state
    if not isinstance(parsed_jobs, list):
        _record_node_result(
            state,
            node_name,
            "error",
            "state 缺少 parsed_jobs。",
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state
    if not isinstance(matched_jobs, list):
        _record_node_result(
            state,
            node_name,
            "error",
            "state 缺少 matched_jobs。",
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    jobs_by_id = _job_lookup(parsed_jobs)
    try:
        gap_top_n = int(state.get("gap_top_n", 1))
    except (TypeError, ValueError):
        gap_top_n = 1
    gap_top_n = max(1, min(gap_top_n, 3))
    use_llm_deep_analysis = bool(state.get("use_llm_deep_analysis", False))
    top_matches = sorted(matched_jobs, key=lambda item: item.get("match_score", 0), reverse=True)[:gap_top_n]

    failed_count = 0
    fallback_count = 0

    for match_result in top_matches:
        try:
            _ensure_execution_allowed(state)
        except (WorkflowCancelledError, WorkflowTimeoutError) as exc:
            state["halt_reason"] = str(exc)
            break
        try:
            if not isinstance(match_result, dict):
                raise ValueError("匹配结果必须是 dict。")

            job_id = match_result.get("job_id")
            job = jobs_by_id.get(job_id)
            if job is None:
                raise ValueError(f"未找到 job_id={job_id} 对应的 JobPosting。")

            if not (state.get("api_available", True) and use_llm_deep_analysis):
                gap_results.append(
                    {
                        "job_id": job_id,
                        "title": job.get("title"),
                        "company": job.get("company"),
                        "gaps": _fallback_gaps(match_result),
                    }
                )
                continue

            try:
                messages = build_gap_analysis_prompt(candidate_profile, job, match_result)
                gap_payload = _call_llm_json_with_retry(state, node_name, messages)
                gaps = _validate_gap_payload(gap_payload)
                gap_results.append(
                    {
                        "job_id": job_id,
                        "title": job.get("title"),
                        "company": job.get("company"),
                        "gaps": gaps,
                    }
                )
            except Exception as exc:
                fallback_count += 1
                gap_results.append(
                    {
                        "job_id": job_id,
                        "title": job.get("title"),
                        "company": job.get("company"),
                        "gaps": _fallback_gaps(match_result),
                    }
                )
                _append_trace(
                    state,
                    node_name,
                    "warning",
                    f"岗位 {job_id} 的差距分析 LLM 调用失败，已使用规则兜底：{exc}",
                    input_count=1,
                    output_count=1,
                    fallback_used=True,
                )
        except Exception as exc:
            failed_count += 1
            job_id = match_result.get("job_id", "<未知岗位>") if isinstance(match_result, dict) else "<未知岗位>"
            _append_trace(
                state,
                node_name,
                "error",
                f"岗位 {job_id} 差距分析失败：{exc}",
                input_count=1,
                output_count=0,
            )
            continue

    if not gap_results and top_matches:
        _record_node_result(
            state,
            node_name,
            "error",
            "Top 岗位均未能生成差距分析。",
            input_count=len(top_matches),
            output_count=0,
            gap_top_n=gap_top_n,
            use_llm=use_llm_deep_analysis,
            failed_count=failed_count,
            fallback_count=fallback_count,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    node_status = "success"
    if failed_count > 0 or fallback_count > 0:
        node_status = "partial"

    _record_node_result(
        state,
        node_name,
        node_status,
        f"已为 {len(top_matches)} 个 Top 岗位生成 {len(gap_results)} 组差距分析。",
        input_count=len(top_matches),
        output_count=len(gap_results),
        gap_top_n=gap_top_n,
        use_llm=use_llm_deep_analysis,
        failed_count=failed_count,
        fallback_count=fallback_count,
        retry_count=_get_retry_count(state, node_name),
        **_node_observability(started_at, token_usage_before),
    )
    return state

def resume_suggestion_node(state: dict) -> dict:
    """Generate resume suggestions for the top jobs with gap analysis."""
    node_name = "resume_suggestion_node"
    started_at = time.perf_counter()
    can_continue, token_usage_before = _begin_node(state, node_name, started_at)
    if not can_continue:
        return state
    resume_suggestions: list[dict[str, Any]] = []
    state["resume_suggestions"] = resume_suggestions

    candidate_profile = state.get("candidate_profile")
    parsed_jobs = state.get("parsed_jobs")
    gaps = state.get("gaps")

    if not isinstance(candidate_profile, dict):
        _record_node_result(
            state,
            node_name,
            "error",
            "state 缺少 candidate_profile。",
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state
    if not isinstance(parsed_jobs, list):
        _record_node_result(
            state,
            node_name,
            "error",
            "state 缺少 parsed_jobs。",
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state
    if not isinstance(gaps, list):
        _record_node_result(
            state,
            node_name,
            "error",
            "state 缺少 gaps。",
            input_count=0,
            output_count=0,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    jobs_by_id = _job_lookup(parsed_jobs)
    try:
        resume_top_n = int(state.get("resume_top_n", 1))
    except (TypeError, ValueError):
        resume_top_n = 1
    resume_top_n = max(1, min(resume_top_n, 3))
    use_llm_deep_analysis = bool(state.get("use_llm_deep_analysis", False))
    top_gap_results = gaps[:resume_top_n]

    failed_count = 0
    fallback_count = 0

    for gap_result in top_gap_results:
        try:
            _ensure_execution_allowed(state)
        except (WorkflowCancelledError, WorkflowTimeoutError) as exc:
            state["halt_reason"] = str(exc)
            break
        try:
            if not isinstance(gap_result, dict):
                raise ValueError("差距分析结果必须是 dict。")

            job_id = gap_result.get("job_id")
            job = jobs_by_id.get(job_id)
            if job is None:
                raise ValueError(f"未找到 job_id={job_id} 对应的 JobPosting。")

            job_gaps = gap_result.get("gaps", [])
            if not isinstance(job_gaps, list):
                raise ValueError("差距分析结果中的 gaps 必须是列表。")

            if not (state.get("api_available", True) and use_llm_deep_analysis):
                resume_suggestions.append(
                    {
                        "job_id": job_id,
                        "title": job.get("title"),
                        "company": job.get("company"),
                        "suggestions": _fallback_resume_suggestions(job_gaps),
                    }
                )
                continue

            try:
                messages = build_resume_suggestion_prompt(candidate_profile, job, job_gaps)
                suggestion_payload = _call_llm_json_with_retry(state, node_name, messages)
                suggestions = _validate_resume_suggestion_payload(suggestion_payload)
                resume_suggestions.append(
                    {
                        "job_id": job_id,
                        "title": job.get("title"),
                        "company": job.get("company"),
                        "suggestions": suggestions,
                    }
                )
            except Exception as exc:
                fallback_count += 1
                resume_suggestions.append(
                    {
                        "job_id": job_id,
                        "title": job.get("title"),
                        "company": job.get("company"),
                        "suggestions": _fallback_resume_suggestions(job_gaps),
                    }
                )
                _append_trace(
                    state,
                    node_name,
                    "warning",
                    f"岗位 {job_id} 的简历建议 LLM 调用失败，已使用规则兜底：{exc}",
                    input_count=1,
                    output_count=1,
                    fallback_used=True,
                )
        except Exception as exc:
            failed_count += 1
            job_id = gap_result.get("job_id", "<未知岗位>") if isinstance(gap_result, dict) else "<未知岗位>"
            _append_trace(
                state,
                node_name,
                "error",
                f"岗位 {job_id} 简历建议生成失败：{exc}",
                input_count=1,
                output_count=0,
            )
            continue

    if not resume_suggestions and top_gap_results:
        _record_node_result(
            state,
            node_name,
            "error",
            "Top 岗位均未能生成简历建议。",
            input_count=len(top_gap_results),
            output_count=0,
            resume_top_n=resume_top_n,
            use_llm=use_llm_deep_analysis,
            failed_count=failed_count,
            fallback_count=fallback_count,
            retry_count=_get_retry_count(state, node_name),
            **_node_observability(started_at, token_usage_before),
        )
        return state

    node_status = "success"
    if failed_count > 0 or fallback_count > 0:
        node_status = "partial"

    _record_node_result(
        state,
        node_name,
        node_status,
        f"已为 {len(top_gap_results)} 个 Top 岗位生成 {len(resume_suggestions)} 组简历建议。",
        input_count=len(top_gap_results),
        output_count=len(resume_suggestions),
        resume_top_n=resume_top_n,
        use_llm=use_llm_deep_analysis,
        failed_count=failed_count,
        fallback_count=fallback_count,
        retry_count=_get_retry_count(state, node_name),
        **_node_observability(started_at, token_usage_before),
    )
    return state

def human_review_node(state: dict) -> dict:
    """Pause the workflow when JD parsing quality is below the configured threshold."""
    from langgraph.types import interrupt

    node_name = "human_review_node"
    started_at = time.perf_counter()
    token_usage_before = get_token_usage()
    failure_rate = float(state.get("jd_parse_failure_rate") or 0.0)
    state["workflow_status"] = "awaiting_review"
    state["review_required"] = True
    state["review_reason"] = f"JD 解析失败率为 {failure_rate:.1%}，需要人工确认后继续。"
    decision = interrupt(
        {
            "type": "jd_parse_review",
            "message": state["review_reason"],
            "failure_rate": failure_rate,
            "parsed_job_count": len(state.get("parsed_jobs") or []),
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    if approved:
        state["workflow_status"] = "running"
        state["review_required"] = False
        message = "人工审核已通过，工作流继续执行。"
        node_status = "success"
    else:
        state["workflow_status"] = "failed"
        state["halt_reason"] = "人工审核未通过，工作流已终止。"
        message = state["halt_reason"]
        node_status = "error"
    _record_node_result(
        state,
        node_name,
        node_status,
        message,
        input_count=int(state.get("jd_parse_input_count") or 0),
        output_count=len(state.get("parsed_jobs") or []),
        parse_failure_rate=failure_rate,
        approved=approved,
        retry_count=_get_retry_count(state, node_name),
        **_node_observability(started_at, token_usage_before),
    )
    return state

def skip_deep_analysis_node(state: dict) -> dict:
    """Skip expensive analysis when no job reaches the configured score threshold."""
    node_name = "skip_deep_analysis_node"
    started_at = time.perf_counter()
    token_usage_before = get_token_usage()
    matched_jobs = state.get("matched_jobs") if isinstance(state.get("matched_jobs"), list) else []
    best_score = max((float(job.get("match_score", 0)) for job in matched_jobs), default=0.0)
    threshold = float(state.get("min_deep_analysis_score") or 35.0)
    state["gaps"] = []
    state["resume_suggestions"] = []
    state["deep_analysis_skipped"] = True
    state["workflow_status"] = "completed"
    _record_node_result(
        state,
        node_name,
        "success",
        f"最高匹配分 {best_score:.1f} 低于阈值 {threshold:.1f}，已跳过深度分析。",
        input_count=len(matched_jobs),
        output_count=0,
        best_match_score=best_score,
        threshold=threshold,
        retry_count=_get_retry_count(state, node_name),
        **_node_observability(started_at, token_usage_before),
    )
    return state

def halt_workflow_node(state: dict) -> dict:
    """Finish a workflow that cannot proceed and preserve a clear reason."""
    node_name = "halt_workflow_node"
    started_at = time.perf_counter()
    token_usage_before = get_token_usage()
    if state.get("workflow_status") not in {"cancelled", "timed_out"}:
        state["workflow_status"] = "failed"
    reason = str(state.get("halt_reason") or "工作流无法继续执行。")
    state.setdefault("matched_jobs", [])
    state.setdefault("gaps", [])
    state.setdefault("resume_suggestions", [])
    _record_node_result(
        state,
        node_name,
        "error",
        reason,
        input_count=0,
        output_count=0,
        retry_count=_get_retry_count(state, node_name),
        **_node_observability(started_at, token_usage_before),
    )
    return state

def finalize_workflow_node(state: dict) -> dict:
    """Mark a normally completed workflow and emit a final summary trace."""
    node_name = "finalize_workflow_node"
    started_at = time.perf_counter()
    token_usage_before = get_token_usage()
    state["workflow_status"] = "completed"
    state["review_required"] = False
    _record_node_result(
        state,
        node_name,
        "success",
        "JobPilot 工作流执行完成。",
        input_count=len(state.get("matched_jobs") or []),
        output_count=len(state.get("resume_suggestions") or []),
        retry_count=_get_retry_count(state, node_name),
        **_node_observability(started_at, token_usage_before),
    )
    return state

def parse_jd_node(state: dict) -> dict:
    """Backward-compatible alias for jd_parse_node."""
    return jd_parse_node(state)


def extract_candidate_node(state: dict) -> dict:
    """Backward-compatible alias for profile_node."""
    return profile_node(state)


def score_match_node(state: dict) -> dict:
    """Backward-compatible alias for match_score_node."""
    return match_score_node(state)
