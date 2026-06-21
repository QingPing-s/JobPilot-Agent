from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.reranker import rerank_jobs
from src.retriever import build_chroma_store, hybrid_retrieve
from src.schemas import MatchResult
from src.scorer import compute_rule_based_match
from src.tools import generate_job_id, load_jd_files, load_user_profile, save_markdown


EVAL_DIR = ROOT_DIR / "eval"
DEFAULT_CASES_PATH = EVAL_DIR / "eval_cases.json"
DEFAULT_REPORT_PATH = EVAL_DIR / "metrics_report.md"
DEFAULT_PROFILE_PATH = ROOT_DIR / "data" / "user_profile.json"
DEFAULT_JD_FOLDER = ROOT_DIR / "data" / "sample_jds"
DEFAULT_VECTOR_STORE = ROOT_DIR / "data" / "vector_store" / "eval"
QUERY_SKILL_MARKERS = {
    "LangGraph": ["langgraph"],
    "ChromaDB": ["chromadb"],
    "DeepSeek API": ["deepseek"],
    "FastAPI": ["fastapi"],
    "Pydantic": ["pydantic"],
    "Python": ["python"],
    "RAG": ["rag", "检索增强", "文档问答", "document qa", "retrieval"],
    "LLM": ["llm", "大模型"],
    "API integration": ["api"],
    "Prompt Engineering": ["prompt", "提示词"],
    "Structured Outputs": ["structured output", "结构化输出"],
    "Tool Calling": ["tool calling", "工具调用"],
    "Evaluation": ["evaluation", "评测", "评估"],
    "Embeddings": ["embedding", "embeddings", "向量表示"],
    "Vector Databases": ["vector database", "vector databases", "向量数据库"],
    "Pytest": ["pytest", "测试"],
}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _read_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _section_lines(raw_text: str, section_name: str) -> list[str]:
    lines = raw_text.splitlines()
    capture = False
    items = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith(":") and not stripped.startswith("-"):
            capture = stripped[:-1].lower() == section_name.lower()
            continue
        if capture and stripped.startswith("-"):
            items.append(stripped.lstrip("-").strip())
    return items


def _extract_title(raw_text: str, fallback: str) -> str:
    for line in raw_text.splitlines():
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()
    return fallback


def _extract_skills(lines: list[str]) -> list[str]:
    text = " ".join(lines).lower()
    skills = []
    checks = [
        ("LangGraph", "langgraph"),
        ("ChromaDB", "chromadb"),
        ("DeepSeek API", "deepseek"),
        ("FastAPI", "fastapi"),
        ("Pydantic", "pydantic"),
        ("Python", "python"),
        ("RAG", "rag"),
        ("LLM", "llm"),
        ("API integration", "api"),
        ("Prompt Engineering", "prompt"),
        ("Structured Outputs", "structured output"),
        ("Tool Calling", "tool"),
        ("Evaluation", "evaluation"),
        ("Embeddings", "embedding"),
        ("Vector Databases", "vector database"),
        ("Pytest", "pytest"),
    ]
    for skill, marker in checks:
        if marker in text and skill not in skills:
            skills.append(skill)
    return skills


def _parse_jobs_from_sample_jds(jd_folder: Path) -> list[dict]:
    parsed_jobs = []
    for index, jd_file in enumerate(load_jd_files(str(jd_folder)), start=1):
        filename = jd_file["filename"]
        raw_text = jd_file["raw_text"]
        responsibilities = _section_lines(raw_text, "Responsibilities")
        requirement_lines = _section_lines(raw_text, "Requirements")
        preferred_lines = _section_lines(raw_text, "Preferred")

        parsed_jobs.append(
            {
                "job_id": generate_job_id(filename, index),
                "title": _extract_title(raw_text, Path(filename).stem),
                "company": "Sample Company",
                "location": None,
                "employment_type": "Internship",
                "salary": None,
                "responsibilities": responsibilities,
                "required_skills": _extract_skills(requirement_lines),
                "preferred_skills": _extract_skills(preferred_lines),
                "education_requirement": None,
                "experience_requirement": None,
                "source_url": None,
                "raw_text": raw_text,
            }
        )
    return parsed_jobs


def _profile_for_case(base_profile: dict, case: dict) -> dict:
    profile = dict(base_profile)
    target_role = case.get("target_role")
    target_roles = list(_as_list(profile.get("target_roles")))
    if target_role and target_role not in target_roles:
        target_roles.insert(0, target_role)
    profile["target_roles"] = target_roles
    profile["target_role"] = target_role

    query = str(case.get("query") or "")
    skills = list(_as_list(profile.get("skills")))
    query_lower = query.lower()
    for skill, markers in QUERY_SKILL_MARKERS.items():
        if any(marker in query_lower for marker in markers) and skill not in skills:
            skills.append(skill)
    profile["skills"] = skills
    return profile


def _validate_match_result(result: dict) -> bool:
    try:
        if hasattr(MatchResult, "model_validate"):
            MatchResult.model_validate(result)
        else:
            MatchResult.parse_obj(result)
    except Exception:
        return False
    return True


def _run_case(case: dict, profile: dict, jobs: list[dict], top_k: int, vector_store: Path) -> dict:
    case_id = case.get("case_id", "<unknown>")
    errors = []
    relevant_job_ids = set(_as_list(case.get("relevant_job_ids")))
    case_profile = _profile_for_case(profile, case)
    query = f"{case.get('target_role', '')} {case.get('query', '')}".strip()

    try:
        build_chroma_store(jobs, persist_dir=str(vector_store / case_id))
    except Exception as exc:
        errors.append(f"向量库构建失败：{exc}")

    try:
        retrieved_jobs = hybrid_retrieve(query, jobs, top_k=top_k, persist_dir=str(vector_store / case_id))
    except Exception as exc:
        errors.append(f"混合召回失败：{exc}")
        retrieved_jobs = jobs[:top_k]

    try:
        reranked_jobs = rerank_jobs(case_profile, retrieved_jobs)
    except Exception as exc:
        errors.append(f"岗位重排失败：{exc}")
        reranked_jobs = retrieved_jobs

    recommendations = []
    for job in reranked_jobs[:top_k]:
        try:
            match = compute_rule_based_match(case_profile, job)
            match["retrieve_source"] = job.get("retrieve_source", "")
            match["rerank_score"] = job.get("rerank_score")
            match["rerank_reason"] = job.get("rerank_reason", "")
            recommendations.append(match)
        except Exception as exc:
            errors.append(f"岗位 {job.get('job_id', '<未知岗位>')} 规则匹配评分失败：{exc}")

    recommendations.sort(key=lambda item: item.get("match_score", 0), reverse=True)
    top_results = recommendations[:top_k]
    top_ids = [item.get("job_id") for item in top_results]
    hits = len(relevant_job_ids & set(top_ids))
    recall = hits / len(relevant_job_ids) if relevant_job_ids else 0.0
    precision_denominator = min(top_k, len(top_results)) or top_k
    precision = hits / precision_denominator if precision_denominator else 0.0
    hit = 1.0 if hits > 0 else 0.0
    valid_count = sum(1 for item in top_results if _validate_match_result(item))

    return {
        "case_id": case_id,
        "target_role": case.get("target_role", ""),
        "query": case.get("query", ""),
        "relevant_job_ids": sorted(relevant_job_ids),
        "recommendations": top_results,
        "metrics": {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "hit_at_k": hit,
            "average_match_score": _average([item.get("match_score", 0) for item in top_results]),
            "json_valid_count": valid_count,
            "json_total_count": len(top_results),
        },
        "errors": errors,
    }


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_recommendation(item: dict, relevant_ids: list[str]) -> str:
    relevant = "是" if item.get("job_id") in set(relevant_ids) else "否"
    source_label = {
        "both": "向量+关键词",
        "keyword": "关键词",
        "vector": "向量",
    }.get(item.get("retrieve_source", ""), "无")
    return (
        f"- `{item.get('job_id', '')}` | {item.get('title', '')} | "
        f"匹配分={item.get('match_score', 0):.2f} | "
        f"召回来源={source_label} | "
        f"重排分={item.get('rerank_score', '')} | 命中相关岗位={relevant}"
    )


def _build_report(case_results: list[dict], top_k: int) -> str:
    total_cases = len(case_results)
    failures = [case for case in case_results if case["errors"] or case["metrics"]["hit_at_k"] == 0.0]
    json_valid_total = sum(case["metrics"]["json_valid_count"] for case in case_results)
    json_total = sum(case["metrics"]["json_total_count"] for case in case_results)
    successful_cases = sum(1 for case in case_results if not case["errors"])

    recall = _average([case["metrics"]["recall_at_k"] for case in case_results])
    precision = _average([case["metrics"]["precision_at_k"] for case in case_results])
    hit = _average([case["metrics"]["hit_at_k"] for case in case_results])
    average_match_score = _average([case["metrics"]["average_match_score"] for case in case_results])
    json_valid_rate = (json_valid_total / json_total) if json_total else 0.0
    tool_success_rate = (successful_cases / total_cases) if total_cases else 0.0

    lines = [
        "# JobPilot-Agent 评测报告",
        "",
        "## 汇总指标",
        "",
        f"- 总 case 数: {total_cases}",
        f"- Recall@{top_k}: {_percent(recall)}",
        f"- Precision@{top_k}: {_percent(precision)}",
        f"- Hit@{top_k}: {_percent(hit)}",
        f"- 平均匹配分: {average_match_score:.2f}",
        f"- JSON 有效率: {_percent(json_valid_rate)}",
        f"- 工具成功率: {_percent(tool_success_rate)}",
        "",
        "说明：评测默认使用确定性的规则匹配作为兜底路径，因此在没有 DeepSeek API 访问权限时也可以运行。",
        "",
        "## 失败用例",
        "",
    ]

    if not failures:
        lines.append("- 无")
    else:
        for case in failures:
            error_text = "；".join(case["errors"]) if case["errors"] else "Top-K 未命中相关岗位。"
            lines.append(f"- `{case['case_id']}`: {error_text}")

    lines.extend(["", f"## 每个用例的 Top-{top_k} 推荐结果", ""])
    for case in case_results:
        lines.extend(
            [
                f"### {case['case_id']} - {case['target_role']}",
                "",
                f"- 查询: {case['query']}",
                f"- 相关 job_id: {', '.join(case['relevant_job_ids'])}",
                f"- Recall@{top_k}: {_percent(case['metrics']['recall_at_k'])}",
                f"- Precision@{top_k}: {_percent(case['metrics']['precision_at_k'])}",
                f"- Hit@{top_k}: {_percent(case['metrics']['hit_at_k'])}",
                f"- 平均匹配分: {case['metrics']['average_match_score']:.2f}",
                "",
            ]
        )
        if not case["recommendations"]:
            lines.append("- 暂无推荐结果。")
        for recommendation in case["recommendations"]:
            lines.append(_format_recommendation(recommendation, case["relevant_job_ids"]))
        if case["errors"]:
            lines.extend(["", "错误信息："])
            for error in case["errors"]:
                lines.append(f"- {error}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_eval(
    cases_path: Path = DEFAULT_CASES_PATH,
    profile_path: Path = DEFAULT_PROFILE_PATH,
    jd_folder: Path = DEFAULT_JD_FOLDER,
    report_path: Path = DEFAULT_REPORT_PATH,
    vector_store: Path = DEFAULT_VECTOR_STORE,
    top_k: int = 5,
) -> dict:
    cases = _read_cases(cases_path)
    profile = load_user_profile(str(profile_path))
    jobs = _parse_jobs_from_sample_jds(jd_folder)

    case_results = []
    for case in cases:
        try:
            case_results.append(_run_case(case, profile, jobs, top_k=top_k, vector_store=vector_store))
        except Exception as exc:
            case_results.append(
                {
                    "case_id": case.get("case_id", "<unknown>"),
                    "target_role": case.get("target_role", ""),
                    "query": case.get("query", ""),
                    "relevant_job_ids": _as_list(case.get("relevant_job_ids")),
                    "recommendations": [],
                    "metrics": {
                        "recall_at_k": 0.0,
                        "precision_at_k": 0.0,
                        "hit_at_k": 0.0,
                        "average_match_score": 0.0,
                        "json_valid_count": 0,
                    "json_total_count": 0,
                },
                    "errors": [f"case 执行失败：{exc}"],
                }
            )

    report = _build_report(case_results, top_k=top_k)
    save_markdown(report, str(report_path))

    return {
        "cases": case_results,
        "report_path": str(report_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 JobPilot-Agent 的检索和匹配效果。")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="eval_cases.json 路径。")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH), help="用户画像 JSON 路径。")
    parser.add_argument("--jd-folder", default=str(DEFAULT_JD_FOLDER), help="包含 JD .txt 文件的文件夹。")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="评测报告输出路径。")
    parser.add_argument("--top-k", type=int, default=5, help="检索指标的 Top-K 截断值。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_eval(
        cases_path=Path(args.cases),
        profile_path=Path(args.profile),
        jd_folder=Path(args.jd_folder),
        report_path=Path(args.report),
        top_k=args.top_k,
    )
    print(f"评测完成，报告路径：{result['report_path']}")


if __name__ == "__main__":
    main()
