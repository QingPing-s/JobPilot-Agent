from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .tools import save_json, save_markdown
from .trace_logger import TraceLogger

ROOT_DIR = Path(__file__).resolve().parents[1]
PERSISTENT_DATA_DIR = os.getenv("JOBPILOT_DATA_DIR")
OUTPUT_DIR = Path(PERSISTENT_DATA_DIR) / "outputs" if PERSISTENT_DATA_DIR else ROOT_DIR / "outputs"
TRACE_DIR = Path(PERSISTENT_DATA_DIR) / "traces" if PERSISTENT_DATA_DIR else ROOT_DIR / "traces"


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT_DIR / candidate


def _validate_env() -> bool:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_deepseek_api_key":
        print(
            "OPENAI_API_KEY 未配置或仍为占位值。将使用本地规则兜底运行；"
            "需要启用 DeepSeek API 时，请基于 .env.example 创建 .env 并填写真实 key。"
        )
        return False
    return True


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _format_list(items: list) -> str:
    if not items:
        return "- 暂无"
    return "\n".join(f"- {item}" for item in items)


def _job_title(job: dict) -> str:
    title = job.get("title") or "未命名岗位"
    company = job.get("company") or "未知公司"
    return f"{title} - {company}"


def build_final_report_markdown(state: dict, target_role: str) -> str:
    candidate_profile = state.get("candidate_profile", {})
    matched_jobs = _as_list(state.get("matched_jobs"))
    gaps = _as_list(state.get("gaps"))
    resume_suggestions = _as_list(state.get("resume_suggestions"))
    token_usage = state.get("token_usage") if isinstance(state.get("token_usage"), dict) else {}

    candidate_roles = _as_list(candidate_profile.get("target_roles")) if isinstance(candidate_profile, dict) else []
    target_role_text = target_role or ", ".join(candidate_roles) or "未指定"

    lines = [
        "# JobPilot-Agent 最终报告",
        "",
        "## 候选人目标岗位",
        "",
        f"- 当前目标岗位: {target_role_text}",
    ]

    if candidate_roles:
        lines.extend(["- 候选人画像中的目标岗位:", _format_list(candidate_roles)])

    lines.extend(
        [
            "",
            "## Token 消耗",
            "",
            f"- LLM 调用次数: {token_usage.get('calls', 0)}",
            f"- Prompt Tokens: {token_usage.get('prompt_tokens', 0)}",
            f"- Completion Tokens: {token_usage.get('completion_tokens', 0)}",
            f"- Total Tokens: {token_usage.get('total_tokens', 0)}",
        ]
    )

    lines.extend(["", "## Top 5 推荐岗位", ""])
    top_jobs = matched_jobs[:5]
    if not top_jobs:
        lines.append("暂无岗位匹配结果。")
    for index, job in enumerate(top_jobs, start=1):
        lines.extend(
            [
                f"### {index}. {_job_title(job)}",
                "",
                f"- job_id: {job.get('job_id', '')}",
                f"- 匹配分: {job.get('match_score', '')}",
                "- 已匹配技能:",
                _format_list(_as_list(job.get("skill_overlap"))),
                "- 缺失技能:",
                _format_list(_as_list(job.get("missing_skills"))),
                f"- 推荐建议: {job.get('recommendation', '')}",
                "",
            ]
        )

    lines.extend(["## Top 3 岗位的差距分析", ""])
    if not gaps:
        lines.append("暂无差距分析结果。")
    for gap_result in gaps[:3]:
        lines.extend([f"### job_id: {gap_result.get('job_id', '')}", ""])
        job_gaps = _as_list(gap_result.get("gaps"))
        if not job_gaps:
            lines.append("- 暂无差距项。")
        for gap in job_gaps:
            lines.extend(
                [
                    f"- 类型: {gap.get('type', '')}",
                    f"  严重程度: {gap.get('severity', '')}",
                    f"  问题描述: {gap.get('description', '')}",
                    f"  优化建议: {gap.get('suggestion', '')}",
                ]
            )
        lines.append("")

    lines.extend(["## Top 3 岗位的简历优化建议", ""])
    if not resume_suggestions:
        lines.append("暂无简历优化建议。")
    for suggestion_result in resume_suggestions[:3]:
        lines.extend([f"### job_id: {suggestion_result.get('job_id', '')}", ""])
        suggestions = _as_list(suggestion_result.get("suggestions"))
        if not suggestions:
            lines.append("- 暂无建议。")
        for suggestion in suggestions:
            lines.extend(
                [
                    f"- 模块: {suggestion.get('section', '')}",
                    f"  原始问题: {suggestion.get('original_problem', '')}",
                    f"  优化建议: {suggestion.get('suggestion', '')}",
                    f"  改写示例: {suggestion.get('improved_example', '')}",
                ]
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_final_report_data(state: dict, target_role: str) -> dict:
    return {
        "target_role": target_role,
        "token_usage": state.get("token_usage", {}),
        "top_5_jobs": _as_list(state.get("matched_jobs"))[:5],
        "top_3_gap_analysis": _as_list(state.get("gaps"))[:3],
        "top_3_resume_suggestions": _as_list(state.get("resume_suggestions"))[:3],
    }


def write_outputs(
    state: dict,
    target_role: str,
    *,
    output_dir: str | Path | None = None,
    trace_dir: str | Path | None = None,
) -> dict[str, Path]:
    resolved_output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    resolved_trace_dir = Path(trace_dir) if trace_dir is not None else TRACE_DIR
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_trace_dir.mkdir(parents=True, exist_ok=True)

    state["final_report"] = build_final_report_data(state, target_role)

    matched_jobs_path = resolved_output_dir / "matched_jobs.json"
    resume_suggestions_path = resolved_output_dir / "resume_suggestions.json"
    final_report_path = resolved_output_dir / "final_report.md"
    trace_path = resolved_trace_dir / "latest_trace.json"

    save_json(_as_list(state.get("matched_jobs")), str(matched_jobs_path))
    save_json(_as_list(state.get("resume_suggestions")), str(resume_suggestions_path))
    trace_logger = TraceLogger()
    trace_logger.records = _as_list(state.get("trace"))
    trace_logger.save(str(trace_path))
    save_markdown(build_final_report_markdown(state, target_role), str(final_report_path))

    return {
        "matched_jobs": matched_jobs_path,
        "resume_suggestions": resume_suggestions_path,
        "final_report": final_report_path,
        "trace": trace_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 JobPilot-Agent。")
    parser.add_argument("--profile", default="data/user_profile.json", help="用户画像 JSON 路径。")
    parser.add_argument("--jd-folder", default="data/sample_jds", help="本地 JD .txt 文件夹。")
    parser.add_argument("--target-role", default="AI Agent Intern", help="本次运行的目标岗位。")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    api_available = _validate_env()

    profile_path = _resolve_project_path(args.profile)
    jd_folder = _resolve_project_path(args.jd_folder)

    if not profile_path.exists():
        raise SystemExit(f"用户画像文件不存在：{profile_path}")
    if not jd_folder.exists() or not jd_folder.is_dir():
        raise SystemExit(f"JD 文件夹不存在或不是目录：{jd_folder}")

    from .graph import run_jobpilot

    initial_state = {
        "user_profile_path": str(profile_path),
        "jd_folder": str(jd_folder),
        "target_role": args.target_role,
        "api_available": api_available,
    }
    final_state = run_jobpilot(initial_state)
    output_paths = write_outputs(final_state, args.target_role)

    print("JobPilot-Agent 运行完成。")
    print(f"岗位匹配结果: {output_paths['matched_jobs']}")
    print(f"简历优化建议: {output_paths['resume_suggestions']}")
    print(f"最终报告: {output_paths['final_report']}")
    print(f"执行轨迹: {output_paths['trace']}")

    return final_state


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
