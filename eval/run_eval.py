from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.job_recorder import extract_job_record
from src.reranker import rule_based_rerank
from src.retriever import (
    build_chroma_store,
    hybrid_retrieve,
    retrieve_jobs,
    simple_bm25_retrieve,
)
from src.schemas import MatchResult
from src.scorer import compute_rule_based_match
from src.tools import load_jd_files, load_user_profile, save_markdown

EVAL_DIR = ROOT_DIR / "eval"
DEFAULT_CASES_PATH = EVAL_DIR / "eval_cases.json"
DEFAULT_REPORT_PATH = EVAL_DIR / "metrics_report.md"
DEFAULT_PROFILE_PATH = ROOT_DIR / "data" / "user_profile.json"
DEFAULT_SEED_PATH = ROOT_DIR / "data" / "job_seed.json"
DEFAULT_JD_FOLDER = ROOT_DIR / "data" / "sample_jds"
DEFAULT_VECTOR_STORE = ROOT_DIR / "data" / "vector_store" / "eval"
BASELINE_NAMES = (
    "keyword",
    "vector",
    "hybrid_union",
    "hybrid_rrf",
    "hybrid_rrf_rerank",
)
QUERY_SKILL_MARKERS = {
    "LangGraph": ["langgraph"],
    "LangChain": ["langchain"],
    "ChromaDB": ["chromadb", "chroma"],
    "DeepSeek API": ["deepseek"],
    "FastAPI": ["fastapi"],
    "Python": ["python"],
    "RAG": ["rag", "检索增强", "文档问答", "retrieval"],
    "LLM": ["llm", "大模型"],
    "Prompt Engineering": ["prompt", "提示词"],
    "Tool Calling": ["tool calling", "function calling", "工具调用"],
    "Multi-Agent": ["multi-agent", "多智能体"],
    "Agent Memory": ["memory", "记忆"],
    "PyTorch": ["pytorch"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "SQL": ["sql", "mysql", "postgresql"],
}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _read_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("eval_cases.json 必须是 JSON 数组。")
    return [case for case in data if isinstance(case, dict)]


def _record_to_job(raw_text: str, filename: str, source: str) -> dict[str, Any]:
    record = extract_job_record(raw_text, filename=filename, source=source)
    return {
        "job_id": record["job_id"],
        "title": record["title"],
        "company": record["company"],
        "location": record.get("location"),
        "employment_type": "实习",
        "salary": record.get("salary"),
        "responsibilities": _as_list(record.get("responsibilities")),
        "required_skills": _as_list(record.get("required_skills")),
        "preferred_skills": _as_list(record.get("preferred_skills")),
        "education_requirement": record.get("education"),
        "experience_requirement": record.get("duration"),
        "source_url": None,
        "raw_text": raw_text,
    }


def _load_jobs(seed_path: Path, jd_folder: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if seed_path.exists():
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        if isinstance(seed, list):
            for index, item in enumerate(seed, start=1):
                if not isinstance(item, dict):
                    continue
                raw_text = str(item.get("raw_text") or "").strip()
                if not raw_text:
                    continue
                filename = str(item.get("filename") or f"seed_{index:03d}.txt")
                jobs.append(_record_to_job(raw_text, filename, str(item.get("source") or "seed")))
    if jobs:
        return jobs

    for item in load_jd_files(str(jd_folder)):
        jobs.append(_record_to_job(item["raw_text"], item["filename"], "sample_jd"))
    return jobs


def _profile_for_case(base_profile: dict, case: dict) -> dict:
    profile = dict(base_profile)
    target_role = str(case.get("target_role") or "")
    target_roles = list(_as_list(profile.get("target_roles")))
    if target_role and target_role not in target_roles:
        target_roles.insert(0, target_role)
    profile["target_roles"] = target_roles
    profile["target_role"] = target_role

    query = str(case.get("query") or "")
    query_lower = query.lower()
    skills = list(_as_list(profile.get("skills")))
    for skill, markers in QUERY_SKILL_MARKERS.items():
        if any(marker in query_lower for marker in markers) and skill not in skills:
            skills.append(skill)
    profile["skills"] = skills
    return profile


def _validate_match_result(result: dict[str, Any]) -> bool:
    try:
        MatchResult.model_validate(result)
    except Exception:
        return False
    return True


def _rank_metrics(ranked_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    def hits_at(k: int) -> int:
        return len(set(ranked_ids[:k]) & relevant_ids)

    recall_5 = hits_at(5) / len(relevant_ids) if relevant_ids else 0.0
    recall_10 = hits_at(10) / len(relevant_ids) if relevant_ids else 0.0
    precision_5 = hits_at(5) / max(1, min(5, len(ranked_ids)))
    hit_5 = float(hits_at(5) > 0)
    hit_10 = float(hits_at(10) > 0)
    reciprocal_rank = 0.0
    for rank, job_id in enumerate(ranked_ids, start=1):
        if job_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, job_id in enumerate(ranked_ids[:10], start=1)
        if job_id in relevant_ids
    )
    ideal_hits = min(10, len(relevant_ids))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg_10 = dcg / ideal_dcg if ideal_dcg else 0.0
    return {
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "precision_at_5": precision_5,
        "hit_at_5": hit_5,
        "hit_at_10": hit_10,
        "mrr": reciprocal_rank,
        "ndcg_at_10": ndcg_10,
        "top_1_accuracy": float(bool(ranked_ids) and ranked_ids[0] in relevant_ids),
    }


def _retrieve_for_baseline(
    baseline: str,
    query: str,
    profile: dict,
    jobs: list[dict],
    vector_store: Path,
    top_k: int,
) -> list[dict]:
    if baseline == "keyword":
        return simple_bm25_retrieve(query, jobs, top_k=top_k)
    if baseline == "vector":
        return retrieve_jobs(query, top_k=top_k, persist_dir=str(vector_store))
    if baseline == "hybrid_union":
        try:
            vector_results = retrieve_jobs(query, top_k=top_k, persist_dir=str(vector_store))
        except Exception:
            vector_results = []
        keyword_results = simple_bm25_retrieve(query, jobs, top_k=top_k)
        merged = []
        seen = set()
        for item in [*vector_results, *keyword_results]:
            job_id = item.get("job_id")
            if job_id and job_id not in seen:
                seen.add(job_id)
                merged.append(item)
        return merged[:top_k]
    hybrid = hybrid_retrieve(query, jobs, top_k=top_k, persist_dir=str(vector_store))
    if baseline == "hybrid_rrf_rerank":
        return rule_based_rerank(profile, hybrid)[:top_k]
    return hybrid


def _run_baseline(
    baseline: str,
    case: dict,
    profile: dict,
    jobs: list[dict],
    vector_store: Path,
    top_k: int,
) -> dict[str, Any]:
    relevant_ids = set(str(item) for item in _as_list(case.get("relevant_job_ids")))
    query = f"{case.get('target_role', '')} {case.get('query', '')}".strip()
    started = time.perf_counter()
    errors: list[str] = []
    fallback_used = False
    try:
        results = _retrieve_for_baseline(baseline, query, profile, jobs, vector_store, top_k)
    except Exception as exc:
        errors.append(f"{baseline} 检索失败：{exc}")
        results = []

    recommendations = []
    json_valid_count = 0
    for job in results[:top_k]:
        try:
            match = compute_rule_based_match(profile, job)
            match["retrieve_source"] = job.get("retrieve_source", "")
            match["vector_rank"] = job.get("vector_rank")
            match["keyword_rank"] = job.get("keyword_rank")
            match["hybrid_score"] = job.get("hybrid_score")
            match["rerank_score"] = job.get("rerank_score")
            recommendations.append(match)
            json_valid_count += int(_validate_match_result(match))
            backend = (job.get("_retrieval") or {}).get("backend")
            fallback_used = fallback_used or (baseline == "vector" and backend == "simple")
        except Exception as exc:
            errors.append(f"岗位 {job.get('job_id', '<unknown>')} 评分失败：{exc}")

    ranked_ids = [str(item.get("job_id") or "") for item in recommendations]
    metrics = _rank_metrics(ranked_ids, relevant_ids)
    metrics.update(
        {
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "json_valid_count": json_valid_count,
            "json_total_count": len(recommendations),
            "fallback_used": float(fallback_used),
            "average_match_score": statistics.fmean(
                [float(item.get("match_score", 0)) for item in recommendations]
            )
            if recommendations
            else 0.0,
        }
    )
    return {
        "baseline": baseline,
        "recommendations": recommendations,
        "metrics": metrics,
        "errors": errors,
    }


def _run_case(
    case: dict,
    base_profile: dict,
    jobs: list[dict],
    vector_store: Path,
    top_k: int,
) -> dict[str, Any]:
    profile = _profile_for_case(base_profile, case)
    baseline_results: dict[str, Any] = {}
    for baseline in BASELINE_NAMES:
        try:
            baseline_results[baseline] = _run_baseline(
                baseline,
                case,
                profile,
                jobs,
                vector_store,
                top_k,
            )
        except Exception as exc:
            baseline_results[baseline] = {
                "baseline": baseline,
                "recommendations": [],
                "metrics": {
                    **_rank_metrics([], set(_as_list(case.get("relevant_job_ids")))),
                    "latency_ms": 0.0,
                    "json_valid_count": 0,
                    "json_total_count": 0,
                    "fallback_used": 1.0,
                    "average_match_score": 0.0,
                },
                "errors": [f"case 执行失败：{exc}"],
            }
    return {
        "case_id": str(case.get("case_id") or "<unknown>"),
        "target_role": str(case.get("target_role") or ""),
        "query": str(case.get("query") or ""),
        "relevant_job_ids": [str(item) for item in _as_list(case.get("relevant_job_ids"))],
        "label_source": str(case.get("label_source") or "unspecified"),
        "baselines": baseline_results,
    }


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, math.ceil(0.95 * len(sorted_values)) - 1)
    return float(sorted_values[index])


def _aggregate(case_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    metric_names = (
        "recall_at_5",
        "recall_at_10",
        "precision_at_5",
        "hit_at_5",
        "hit_at_10",
        "mrr",
        "ndcg_at_10",
        "top_1_accuracy",
        "average_match_score",
        "fallback_used",
    )
    for baseline in BASELINE_NAMES:
        metrics = [case["baselines"][baseline]["metrics"] for case in case_results]
        latencies = [float(item["latency_ms"]) for item in metrics]
        json_valid = sum(int(item["json_valid_count"]) for item in metrics)
        json_total = sum(int(item["json_total_count"]) for item in metrics)
        errors = sum(bool(case["baselines"][baseline]["errors"]) for case in case_results)
        values = {
            name: statistics.fmean(float(item[name]) for item in metrics) if metrics else 0.0
            for name in metric_names
        }
        values.update(
            {
                "average_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
                "p95_latency_ms": _percentile_95(latencies),
                "json_valid_rate": json_valid / json_total if json_total else 0.0,
                "tool_success_rate": 1.0 - (errors / len(case_results) if case_results else 0.0),
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "estimated_cost_usd": 0.0,
            }
        )
        aggregate[baseline] = values
    return aggregate


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _build_report(
    case_results: list[dict[str, Any]],
    aggregate: dict[str, dict[str, float]],
    corpus_size: int,
) -> str:
    lines = [
        "# JobPilot-Agent 离线评测报告",
        "",
        f"- 评测 case 数：{len(case_results)}",
        f"- 岗位语料规模：{corpus_size}",
        "- 评测模式：确定性检索与规则重排，不调用真实 LLM",
        "",
        "## 五组基线与融合消融对比",
        "",
        "| 基线 | Recall@5 | Recall@10 | Precision@5 | Hit@5 | MRR | NDCG@10 | Top1 | 平均延迟 | P95 延迟 | 降级率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for baseline in BASELINE_NAMES:
        metric = aggregate[baseline]
        lines.append(
            f"| {baseline} | {_percent(metric['recall_at_5'])} | "
            f"{_percent(metric['recall_at_10'])} | {_percent(metric['precision_at_5'])} | "
            f"{_percent(metric['hit_at_5'])} | {metric['mrr']:.3f} | "
            f"{metric['ndcg_at_10']:.3f} | {_percent(metric['top_1_accuracy'])} | "
            f"{metric['average_latency_ms']:.1f} ms | {metric['p95_latency_ms']:.1f} ms | "
            f"{_percent(metric['fallback_used'])} |"
        )

    best = aggregate["hybrid_rrf_rerank"]
    lines.extend(
        [
            "",
            "## 结构化输出与成本",
            "",
            f"- JSON 有效率：{_percent(best['json_valid_rate'])}",
            f"- 工具成功率：{_percent(best['tool_success_rate'])}",
            f"- Prompt Tokens：{int(best['prompt_tokens'])}",
            f"- Completion Tokens：{int(best['completion_tokens'])}",
            f"- 估算成本：${best['estimated_cost_usd']:.4f}",
            "- 说明：离线基线不调用 LLM，因此 Token 和成本为 0；线上运行成本由 Trace 单独统计。",
            "",
            "## 失败 Case",
            "",
        ]
    )
    failures = [
        case
        for case in case_results
        if case["baselines"]["hybrid_rrf_rerank"]["metrics"]["hit_at_10"] == 0
        or case["baselines"]["hybrid_rrf_rerank"]["errors"]
    ]
    if not failures:
        lines.append("- 无")
    for case in failures:
        errors = "；".join(case["baselines"]["hybrid_rrf_rerank"]["errors"])
        lines.append(f"- `{case['case_id']}`：{errors or 'Top 10 未命中人工标注相关岗位。'}")

    lines.extend(["", "## 每个 Case 的 Hybrid + Rerank Top 10", ""])
    for case in case_results:
        result = case["baselines"]["hybrid_rrf_rerank"]
        lines.extend(
            [
                f"### {case['case_id']} - {case['target_role']}",
                "",
                f"- Query：{case['query']}",
                f"- Relevant：{', '.join(case['relevant_job_ids'])}",
                f"- Label：{case['label_source']}",
            ]
        )
        for rank, item in enumerate(result["recommendations"][:10], start=1):
            hit = "命中" if item.get("job_id") in set(case["relevant_job_ids"]) else ""
            lines.append(
                f"- {rank}. `{item.get('job_id', '')}` {item.get('title', '')} "
                f"| match={item.get('match_score', 0):.1f} "
                f"| hybrid={item.get('hybrid_score') or 0:.6f} "
                f"| rerank={item.get('rerank_score') or 0:.1f} {hit}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_eval(
    cases_path: Path = DEFAULT_CASES_PATH,
    profile_path: Path = DEFAULT_PROFILE_PATH,
    jd_folder: Path = DEFAULT_JD_FOLDER,
    report_path: Path = DEFAULT_REPORT_PATH,
    vector_store: Path = DEFAULT_VECTOR_STORE,
    seed_path: Path = DEFAULT_SEED_PATH,
    top_k: int = 10,
    *,
    allow_small_eval: bool = False,
) -> dict[str, Any]:
    cases = _read_cases(cases_path)
    if len(cases) < 50 and not allow_small_eval:
        raise ValueError(f"正式评测至少需要 50 个 case，当前只有 {len(cases)} 个。")
    profile = load_user_profile(str(profile_path))
    jobs = _load_jobs(seed_path, jd_folder)
    if not jobs:
        raise ValueError("没有可用于评测的岗位语料。")

    build_chroma_store(jobs, persist_dir=str(vector_store))
    case_results = [
        _run_case(case, profile, jobs, vector_store, max(10, top_k))
        for case in cases
    ]
    aggregate = _aggregate(case_results)
    report = _build_report(case_results, aggregate, corpus_size=len(jobs))
    save_markdown(report, str(report_path))
    metrics_path = report_path.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "case_count": len(cases),
                "corpus_size": len(jobs),
                "baselines": aggregate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "cases": case_results,
        "baselines": aggregate,
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 JobPilot-Agent 四基线离线评测。")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--jd-folder", default=str(DEFAULT_JD_FOLDER))
    parser.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--vector-store", default=str(DEFAULT_VECTOR_STORE))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_eval(
        cases_path=Path(args.cases),
        profile_path=Path(args.profile),
        jd_folder=Path(args.jd_folder),
        seed_path=Path(args.seed),
        report_path=Path(args.report),
        vector_store=Path(args.vector_store),
        top_k=args.top_k,
    )
    print(f"评测完成：{result['report_path']}")


if __name__ == "__main__":
    main()
