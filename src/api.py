from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .graph import run_jobpilot
from .job_recorder import DEFAULT_RECORDS_PATH, save_job_records
from .job_store import DEFAULT_DB_PATH, export_active_jobs_to_jd_folder, list_jobs, soft_delete_job, upsert_jobs
from .main import ROOT_DIR, build_final_report_markdown, write_outputs
from .schemas import CandidateProfile
from .tools import save_json


RUNS_DIR = ROOT_DIR / "outputs" / "api_runs"
DEFAULT_PROFILE_PATH = ROOT_DIR / "data" / "user_profile.json"
DEFAULT_JD_FOLDER = ROOT_DIR / "data" / "sample_jds"
DEFAULT_JOB_DB_PATH = ROOT_DIR / DEFAULT_DB_PATH
LATEST_TRACE_PATH = ROOT_DIR / "traces" / "latest_trace.json"
LATEST_REPORT_PATH = ROOT_DIR / "outputs" / "final_report.md"


class RunJobPilotRequest(BaseModel):
    target_role: str = Field(default="AI Agent Intern", description="Target internship role.")
    user_profile_text: str | None = Field(default=None, description="Raw resume, project, or intro text.")
    user_profile_json: dict[str, Any] | None = Field(default=None, description="Structured CandidateProfile JSON.")
    jd_texts: list[str] = Field(default_factory=list, description="Raw JD texts read by the frontend.")
    jd_filenames: list[str] = Field(default_factory=list, description="Optional JD filenames matching jd_texts.")
    use_job_library: bool = Field(default=False, description="Whether to use active jobs from the SQLite job library.")
    retrieval_top_k: int = Field(default=10, ge=1, le=50, description="Number of jobs to retrieve before rerank.")
    use_llm_rerank: bool = Field(default=False, description="Whether to use LLM reranking.")
    use_llm_match_scoring: bool = Field(default=False, description="Whether to use LLM match explanations.")


class RunJobPilotResponse(BaseModel):
    run_id: str
    api_available: bool
    matched_jobs: list[dict[str, Any]]
    gaps: list[dict[str, Any]]
    resume_suggestions: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    final_report: dict[str, Any]
    final_report_markdown: str
    output_paths: dict[str, str]


class RecordJobsRequest(BaseModel):
    jd_texts: list[str] = Field(default_factory=list, description="Raw JD texts to save into the local job library.")
    jd_filenames: list[str] = Field(default_factory=list, description="Optional filenames matching jd_texts.")
    source: str = Field(default="frontend", description="Where these jobs were captured from.")


class RecordJobsResponse(BaseModel):
    saved_count: int
    jobs: list[dict[str, Any]]
    record_path: str
    jd_folder: str
    db_path: str


class JobsListResponse(BaseModel):
    jobs: list[dict[str, Any]]
    count: int
    db_path: str


class DeleteJobResponse(BaseModel):
    deleted: bool
    job_id: str


def _api_available() -> bool:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(api_key and api_key != "your_deepseek_api_key")


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _make_run_dir() -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _write_profile_input(request: RunJobPilotRequest, run_dir: Path) -> tuple[str | None, str | None]:
    if request.user_profile_json:
        if hasattr(CandidateProfile, "model_validate"):
            profile = CandidateProfile.model_validate(request.user_profile_json)
            profile_data = profile.model_dump()
        else:
            profile = CandidateProfile.parse_obj(request.user_profile_json)
            profile_data = profile.dict()
        profile_path = run_dir / "user_profile.json"
        save_json(profile_data, str(profile_path))
        return str(profile_path), None

    profile_text = (request.user_profile_text or "").strip()
    if profile_text:
        return None, profile_text

    return str(DEFAULT_PROFILE_PATH), None


def _safe_jd_filename(index: int, provided_name: str | None) -> str:
    stem = Path(provided_name or f"job_{index:02d}.txt").stem
    safe_stem = "".join(char if char.isalnum() else "_" for char in stem).strip("_")
    if not safe_stem:
        safe_stem = f"job_{index:02d}"
    return f"{safe_stem}.txt"


def _write_jd_inputs(request: RunJobPilotRequest, run_dir: Path) -> str:
    jd_texts = [text.strip() for text in request.jd_texts if text and text.strip()]
    if request.use_job_library and not jd_texts:
        jd_dir = run_dir / "job_library_jds"
        exported_paths = export_active_jobs_to_jd_folder(jd_dir, db_path=DEFAULT_JOB_DB_PATH)
        if not exported_paths:
            raise ValueError("岗位库为空，请先保存 JD 到岗位库，或切换为手动输入 JD。")
        return str(jd_dir)

    if not jd_texts:
        return str(DEFAULT_JD_FOLDER)

    jd_dir = run_dir / "jds"
    jd_dir.mkdir(parents=True, exist_ok=True)
    for index, jd_text in enumerate(jd_texts, start=1):
        provided_name = request.jd_filenames[index - 1] if index <= len(request.jd_filenames) else None
        filename = _safe_jd_filename(index, provided_name)
        (jd_dir / filename).write_text(jd_text, encoding="utf-8")
    return str(jd_dir)


def _build_initial_state(request: RunJobPilotRequest, run_dir: Path, api_available: bool) -> dict[str, Any]:
    profile_path, profile_text = _write_profile_input(request, run_dir)
    jd_folder = _write_jd_inputs(request, run_dir)

    state: dict[str, Any] = {
        "target_role": request.target_role,
        "jd_folder": jd_folder,
        "retrieval_top_k": request.retrieval_top_k,
        "use_llm_rerank": request.use_llm_rerank,
        "use_llm_match_scoring": request.use_llm_match_scoring,
        "api_available": api_available,
        "vector_store_dir": str(run_dir / "vector_store"),
    }
    if profile_text:
        state["user_profile_text"] = profile_text
    elif profile_path:
        state["user_profile_path"] = profile_path
    return state


app = FastAPI(
    title="JobPilot-Agent API",
    description="FastAPI wrapper for the JobPilot LangGraph workflow.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "api_available": _api_available(),
        "project_root": str(ROOT_DIR),
    }


@app.post("/api/run-jobpilot", response_model=RunJobPilotResponse)
def run_jobpilot_api(request: RunJobPilotRequest) -> RunJobPilotResponse:
    run_id, run_dir = _make_run_dir()
    api_available = _api_available()

    try:
        initial_state = _build_initial_state(request, run_dir, api_available)
        final_state = run_jobpilot(initial_state)
        output_paths = write_outputs(final_state, request.target_role)
        final_report_markdown = build_final_report_markdown(final_state, request.target_role)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"JobPilot 运行失败：{exc}") from exc

    return RunJobPilotResponse(
        run_id=run_id,
        api_available=api_available,
        matched_jobs=_as_list(final_state.get("matched_jobs")),
        gaps=_as_list(final_state.get("gaps")),
        resume_suggestions=_as_list(final_state.get("resume_suggestions")),
        trace=_as_list(final_state.get("trace")),
        final_report=final_state.get("final_report", {}),
        final_report_markdown=final_report_markdown,
        output_paths={key: str(value) for key, value in output_paths.items()},
    )


@app.post("/api/record-jobs", response_model=RecordJobsResponse)
def record_jobs_api(request: RecordJobsRequest) -> RecordJobsResponse:
    jd_texts = [text.strip() for text in request.jd_texts if isinstance(text, str) and text.strip()]
    if not jd_texts:
        raise HTTPException(status_code=400, detail="没有可保存的岗位 JD 文本。")

    try:
        db_jobs = upsert_jobs(
            jd_texts=jd_texts,
            jd_filenames=request.jd_filenames,
            source=request.source,
            db_path=DEFAULT_JOB_DB_PATH,
        )
        jobs = save_job_records(
            jd_texts=jd_texts,
            jd_filenames=request.jd_filenames,
            source=request.source,
            records_path=ROOT_DIR / DEFAULT_RECORDS_PATH,
            jd_dir=DEFAULT_JD_FOLDER,
        )
        job_file_by_id = {job.get("job_id"): job for job in jobs}
        for job in db_jobs:
            file_job = job_file_by_id.get(job.get("job_id"))
            if file_job:
                job["record_path"] = file_job.get("record_path")
                job["jd_file_path"] = file_job.get("jd_file_path")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存岗位失败：{exc}") from exc

    return RecordJobsResponse(
        saved_count=len(db_jobs),
        jobs=db_jobs,
        record_path=str(ROOT_DIR / DEFAULT_RECORDS_PATH),
        jd_folder=str(DEFAULT_JD_FOLDER),
        db_path=str(DEFAULT_JOB_DB_PATH),
    )


@app.get("/api/jobs", response_model=JobsListResponse)
def list_jobs_api(include_inactive: bool = False) -> JobsListResponse:
    try:
        jobs = list_jobs(db_path=DEFAULT_JOB_DB_PATH, include_inactive=include_inactive)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取岗位库失败：{exc}") from exc
    return JobsListResponse(jobs=jobs, count=len(jobs), db_path=str(DEFAULT_JOB_DB_PATH))


@app.delete("/api/jobs/{job_id}", response_model=DeleteJobResponse)
def delete_job_api(job_id: str) -> DeleteJobResponse:
    try:
        deleted = soft_delete_job(job_id, db_path=DEFAULT_JOB_DB_PATH)
        (DEFAULT_JD_FOLDER / f"{job_id}.txt").unlink(missing_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除岗位失败：{exc}") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"岗位不存在或已删除：{job_id}")
    return DeleteJobResponse(deleted=True, job_id=job_id)


@app.get("/api/latest-trace")
def latest_trace() -> dict[str, Any]:
    if not LATEST_TRACE_PATH.exists():
        raise HTTPException(status_code=404, detail="尚未生成最新 trace 文件。")
    try:
        return {"trace": json.loads(LATEST_TRACE_PATH.read_text(encoding="utf-8"))}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"最新 trace JSON 格式无效：{exc}") from exc


@app.get("/api/latest-report")
def latest_report() -> dict[str, Any]:
    if not LATEST_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="尚未生成最新 final_report 文件。")
    return {"markdown": LATEST_REPORT_PATH.read_text(encoding="utf-8")}
