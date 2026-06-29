from __future__ import annotations

import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audit_logger import log_audit_event
from .graph import resume_jobpilot, run_jobpilot
from .job_recorder import save_job_records
from .job_store import (
    DEFAULT_DB_PATH,
    list_jobs,
    list_parsed_jobs,
    restore_job,
    soft_delete_job,
    upsert_jobs,
)
from .main import OUTPUT_DIR, ROOT_DIR, TRACE_DIR, build_final_report_markdown, write_outputs
from .retriever import (
    BACKEND_FILE,
    EMBEDDING_MODEL_NAME,
    INDEX_VERSION,
    STORE_FILE,
    build_chroma_store,
)
from .run_control import events_since
from .run_manager import TERMINAL_STATUSES, RunManager, RunStore
from .schemas import CandidateProfile
from .security import (
    AuthUser,
    LoginRequest,
    TokenResponse,
    admin_user,
    authenticate,
    create_access_token,
    current_user,
    enforce_run_rate_limit,
    get_security_settings,
)
from .tools import save_json

PERSISTENT_DATA_DIR = Path(os.getenv("JOBPILOT_DATA_DIR", ROOT_DIR / "data"))
RUNS_DIR = OUTPUT_DIR / "api_runs"
DEFAULT_PROFILE_PATH = ROOT_DIR / "data" / "user_profile.json"
DEFAULT_JD_FOLDER = PERSISTENT_DATA_DIR / "sample_jds"
DEFAULT_JOB_DB_PATH = ROOT_DIR / DEFAULT_DB_PATH
LATEST_TRACE_PATH = TRACE_DIR / "latest_trace.json"
LATEST_REPORT_PATH = OUTPUT_DIR / "final_report.md"
JOB_RECORDS_PATH = PERSISTENT_DATA_DIR / "jobs_csv" / "job_records.jsonl"
# Backward-compatible name used by tests and existing integrations.
DEFAULT_RECORDS_PATH = JOB_RECORDS_PATH
JOB_SEED_PATH = ROOT_DIR / "data" / "job_seed.json"
FRONTEND_DIST = Path(os.getenv("JOBPILOT_FRONTEND_DIST", ROOT_DIR / "frontend" / "dist"))
CHECKPOINT_PATH = PERSISTENT_DATA_DIR / "jobpilot_checkpoints.sqlite"
RUN_DB_PATH = PERSISTENT_DATA_DIR / "jobpilot_runs.db"
RUN_MANAGER = RunManager(store=RunStore(RUN_DB_PATH))


class RunJobPilotRequest(BaseModel):
    target_role: str = Field(default="AI Agent Intern", description="Target internship role.")
    user_profile_text: str | None = Field(default=None, description="Raw resume, project, or intro text.")
    user_profile_json: dict[str, Any] | None = Field(default=None, description="Structured CandidateProfile JSON.")
    jd_texts: list[str] = Field(default_factory=list, description="Raw JD texts read by the frontend.")
    jd_filenames: list[str] = Field(default_factory=list, description="Optional JD filenames matching jd_texts.")
    use_job_library: bool = Field(default=False, description="Whether to use active jobs from the SQLite job library.")
    retrieval_top_k: int = Field(default=20, ge=1, le=50, description="Number of jobs to retrieve before rerank.")
    use_llm_rerank: bool = Field(default=False, description="Whether to use LLM reranking.")
    use_llm_match_scoring: bool = Field(default=False, description="Whether to use LLM match explanations.")
    deep_analysis: bool = Field(default=False, description="Whether to use LLM gap/resume analysis for the top 3 jobs.")
    timeout_seconds: float = Field(default=300.0, ge=30.0, le=1800.0, description="Cooperative run timeout.")
    allow_cache: bool = Field(default=True, description="Reuse an identical completed run owned by the same user.")
    llm_node_max_retries: int = Field(default=1, ge=0, le=3, description="Node-level LLM retry limit.")
    min_deep_analysis_score: float = Field(default=35.0, ge=0.0, le=100.0)
    require_human_review_on_parse_failure: bool = False
    jd_parse_review_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class RunJobPilotResponse(BaseModel):
    run_id: str
    api_available: bool
    matched_jobs: list[dict[str, Any]]
    gaps: list[dict[str, Any]]
    resume_suggestions: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    token_usage: dict[str, int | float] = Field(default_factory=dict)
    final_report: dict[str, Any]
    final_report_markdown: str
    output_paths: dict[str, str]
    workflow_status: str = "completed"
    review_required: bool = False
    review_reason: str | None = None
    checkpoint_backend: str | None = None


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
    vector_store_dir: str
    index_refreshed: bool
    index_warning: str | None = None


class JobsListResponse(BaseModel):
    jobs: list[dict[str, Any]]
    count: int
    db_path: str


class DeleteJobResponse(BaseModel):
    deleted: bool
    job_id: str
    index_refreshed: bool
    index_warning: str | None = None


class RestoreJobResponse(BaseModel):
    restored: bool
    job_id: str
    index_refreshed: bool
    index_warning: str | None = None


class AsyncRunCreatedResponse(BaseModel):
    run_id: str
    status: str
    cache_hit: bool = False
    status_url: str
    events_url: str


class AsyncRunStatusResponse(BaseModel):
    run_id: str
    status: str
    cache_hit: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class ReviewDecisionRequest(BaseModel):
    approved: bool


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


def _default_vector_store_dir() -> Path:
    return PERSISTENT_DATA_DIR / "vector_store"


def _refresh_job_retrieval_store() -> tuple[bool, str | None]:
    """Refresh the persistent retrieval index from active SQLite jobs."""
    try:
        jobs = list_parsed_jobs(db_path=DEFAULT_JOB_DB_PATH)
        build_chroma_store(jobs, persist_dir=str(_default_vector_store_dir()))
        return True, None
    except Exception as exc:
        return False, str(exc)


def _vector_store_needs_refresh() -> bool:
    vector_store_dir = _default_vector_store_dir()
    store_path = vector_store_dir / STORE_FILE
    backend_path = vector_store_dir / BACKEND_FILE
    if not store_path.exists() or not backend_path.exists():
        return True

    if not DEFAULT_JOB_DB_PATH.exists():
        return False

    db_mtime = DEFAULT_JOB_DB_PATH.stat().st_mtime
    return db_mtime > min(store_path.stat().st_mtime, backend_path.stat().st_mtime)


def _initialize_persistent_data() -> None:
    """Seed an empty deployment and create its initial retrieval index."""
    PERSISTENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if list_jobs(db_path=DEFAULT_JOB_DB_PATH):
        if _vector_store_needs_refresh():
            _refresh_job_retrieval_store()
        return
    if not JOB_SEED_PATH.exists():
        return

    seed_records = json.loads(JOB_SEED_PATH.read_text(encoding="utf-8"))
    if not isinstance(seed_records, list):
        raise ValueError("data/job_seed.json must contain a JSON array.")

    jd_texts = []
    jd_filenames = []
    for record in seed_records:
        if not isinstance(record, dict):
            continue
        raw_text = str(record.get("raw_text") or "").strip()
        if not raw_text:
            continue
        jd_texts.append(raw_text)
        jd_filenames.append(str(record.get("filename") or f"seed_{len(jd_texts):03d}.txt"))

    if jd_texts:
        upsert_jobs(
            jd_texts=jd_texts,
            jd_filenames=jd_filenames,
            source="deployment_seed",
            db_path=DEFAULT_JOB_DB_PATH,
        )
        _refresh_job_retrieval_store()


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
    jd_texts = [text.strip() for text in request.jd_texts if text and text.strip()]
    parsed_jobs: list[dict[str, Any]] | None = None
    skip_jd_parse = False

    if request.use_job_library and not jd_texts:
        parsed_jobs = list_parsed_jobs(db_path=DEFAULT_JOB_DB_PATH)
        if not parsed_jobs:
            raise ValueError("岗位库为空，请先保存 JD 到岗位库，或切换为手动输入 JD。")
        jd_folder = ""
        skip_jd_parse = True
    else:
        jd_folder = _write_jd_inputs(request, run_dir)

    state: dict[str, Any] = {
        "target_role": request.target_role,
        "jd_folder": jd_folder,
        "retrieval_top_k": request.retrieval_top_k,
        "rerank_top_k": 10,
        "llm_rerank_top_n": 5,
        "llm_match_top_n": 3,
        "gap_top_n": 3 if request.deep_analysis else 1,
        "resume_top_n": 3 if request.deep_analysis else 1,
        "use_llm_rerank": request.use_llm_rerank,
        "use_llm_match_scoring": request.use_llm_match_scoring,
        "use_llm_deep_analysis": request.deep_analysis,
        "deep_analysis": request.deep_analysis,
        "llm_node_max_retries": request.llm_node_max_retries,
        "min_deep_analysis_score": request.min_deep_analysis_score,
        "require_human_review_on_parse_failure": request.require_human_review_on_parse_failure,
        "jd_parse_review_threshold": request.jd_parse_review_threshold,
        "api_available": api_available,
        "vector_store_dir": str(_default_vector_store_dir() if parsed_jobs is not None else run_dir / "vector_store"),
    }
    if parsed_jobs is not None:
        state["parsed_jobs"] = parsed_jobs
        state["skip_jd_parse"] = skip_jd_parse
        state["job_source"] = "sqlite_job_library"
    if profile_text:
        state["user_profile_text"] = profile_text
    elif profile_path:
        state["user_profile_path"] = profile_path
    return state


def _request_hash(request: RunJobPilotRequest) -> str:
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    payload.pop("timeout_seconds", None)
    payload.pop("allow_cache", None)
    if request.use_job_library:
        library_records = list_jobs(db_path=DEFAULT_JOB_DB_PATH)
        version_payload = [
            {
                "job_id": item.get("job_id"),
                "content_hash": item.get("content_hash"),
                "updated_at": item.get("updated_at"),
            }
            for item in sorted(library_records, key=lambda item: str(item.get("job_id") or ""))
        ]
        payload["job_library_version"] = hashlib.sha256(
            json.dumps(
                version_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _result_payload(
    final_state: dict[str, Any],
    target_role: str,
    output_paths: dict[str, Path],
    *,
    run_id: str,
    api_available: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "api_available": api_available,
        "matched_jobs": _as_list(final_state.get("matched_jobs")),
        "gaps": _as_list(final_state.get("gaps")),
        "resume_suggestions": _as_list(final_state.get("resume_suggestions")),
        "trace": _as_list(final_state.get("trace")),
        "token_usage": final_state.get("token_usage", {}),
        "final_report": final_state.get("final_report", {}),
        "final_report_markdown": build_final_report_markdown(final_state, target_role),
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "workflow_status": final_state.get("workflow_status", "completed"),
        "review_required": bool(final_state.get("review_required")),
        "review_reason": final_state.get("review_reason"),
        "checkpoint_backend": final_state.get("checkpoint_backend"),
    }


def _execute_async_request(
    request: RunJobPilotRequest,
    owner: AuthUser,
    run_id: str,
    deadline_epoch: float | None,
) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    api_available = _api_available()
    initial_state = _build_initial_state(request, run_dir, api_available)
    initial_state["run_id"] = run_id
    if deadline_epoch is not None:
        initial_state["_deadline_epoch"] = deadline_epoch
    final_state = run_jobpilot(
        initial_state,
        thread_id=run_id,
        checkpoint_path=CHECKPOINT_PATH,
    )
    output_paths = write_outputs(
        final_state,
        request.target_role,
        output_dir=run_dir / "outputs",
        trace_dir=run_dir / "traces",
    )
    result = _result_payload(
        final_state,
        request.target_role,
        output_paths,
        run_id=run_id,
        api_available=api_available,
    )
    log_audit_event(
        "run.complete",
        actor_id=owner.user_id,
        role=owner.role,
        resource_type="run",
        resource_id=run_id,
        outcome=result["workflow_status"],
        metadata={
            "matched_jobs": len(result["matched_jobs"]),
            "token_total": result["token_usage"].get("total_tokens", 0),
        },
    )
    return result


def _resume_async_request(
    run_id: str,
    approved: bool,
    owner: AuthUser,
) -> dict[str, Any]:
    final_state = resume_jobpilot(
        run_id,
        approved=approved,
        checkpoint_path=CHECKPOINT_PATH,
    )
    target_role = str(final_state.get("target_role") or "AI Agent Intern")
    run_dir = RUNS_DIR / run_id
    output_paths = write_outputs(
        final_state,
        target_role,
        output_dir=run_dir / "outputs",
        trace_dir=run_dir / "traces",
    )
    result = _result_payload(
        final_state,
        target_role,
        output_paths,
        run_id=run_id,
        api_available=_api_available(),
    )
    log_audit_event(
        "run.review",
        actor_id=owner.user_id,
        role=owner.role,
        resource_type="run",
        resource_id=run_id,
        outcome=result["workflow_status"],
        metadata={"approved": approved},
    )
    return result


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(_initialize_persistent_data)
    yield


app = FastAPI(
    title="JobPilot-Agent API",
    description="FastAPI wrapper for the JobPilot LangGraph workflow.",
    version="0.1.0",
    lifespan=lifespan,
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


@app.post("/api/auth/login", response_model=TokenResponse)
def login(request: LoginRequest) -> TokenResponse:
    if not get_security_settings().auth_enabled:
        raise HTTPException(status_code=400, detail="当前环境未启用登录认证。")
    user = authenticate(request.username, request.password)
    if user is None:
        log_audit_event(
            "auth.login",
            actor_id=request.username,
            role="unknown",
            resource_type="session",
            outcome="denied",
        )
        raise HTTPException(status_code=401, detail="用户名或密码错误。")
    token = create_access_token(user)
    log_audit_event(
        "auth.login",
        actor_id=user.user_id,
        role=user.role,
        resource_type="session",
    )
    return token


@app.get("/api/health")
def health() -> dict[str, Any]:
    security = get_security_settings()
    return {
        "status": "ok",
        "api_available": _api_available(),
        "auth_enabled": security.auth_enabled,
        "project_root": str(ROOT_DIR),
        "data_dir": str(PERSISTENT_DATA_DIR),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "index_version": INDEX_VERSION,
    }


@app.post("/api/runs", response_model=AsyncRunCreatedResponse, status_code=202)
def create_async_run(
    payload: RunJobPilotRequest,
    request: Request,
    user: AuthUser = Depends(current_user),
) -> AsyncRunCreatedResponse:
    enforce_run_rate_limit(request, user)
    request_hash = _request_hash(payload)
    record = RUN_MANAGER.submit(
        user.user_id,
        request_hash,
        lambda run_id, deadline: _execute_async_request(payload, user, run_id, deadline),
        timeout_seconds=payload.timeout_seconds,
        allow_cache=payload.allow_cache,
    )
    run_id = record["run_id"]
    log_audit_event(
        "run.create",
        actor_id=user.user_id,
        role=user.role,
        resource_type="run",
        resource_id=run_id,
        metadata={"cache_hit": bool(record.get("cache_hit"))},
    )
    return AsyncRunCreatedResponse(
        run_id=run_id,
        status=record["status"],
        cache_hit=bool(record.get("cache_hit")),
        status_url=f"/api/runs/{run_id}",
        events_url=f"/api/runs/{run_id}/events",
    )


@app.get("/api/runs/{run_id}", response_model=AsyncRunStatusResponse)
def get_async_run(
    run_id: str,
    user: AuthUser = Depends(current_user),
) -> AsyncRunStatusResponse:
    try:
        record = RUN_MANAGER.get_for_owner(run_id, user.user_id, is_admin=user.is_admin)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="运行记录不存在。") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="无权查看该运行记录。") from exc
    return AsyncRunStatusResponse(**record)


@app.get("/api/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    user: AuthUser = Depends(current_user),
) -> StreamingResponse:
    try:
        RUN_MANAGER.get_for_owner(run_id, user.user_id, is_admin=user.is_admin)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="运行记录不存在。") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="无权查看该运行记录。") from exc

    async def event_stream():
        offset = 0
        last_status = None
        while True:
            record = RUN_MANAGER.get_for_owner(run_id, user.user_id, is_admin=user.is_admin)
            events, offset = events_since(run_id, offset)
            for event in events:
                yield f"event: node\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            if record["status"] != last_status:
                last_status = record["status"]
                status_payload = {"run_id": run_id, "status": last_status}
                yield f"event: status\ndata: {json.dumps(status_payload, ensure_ascii=False)}\n\n"
            if record["status"] in TERMINAL_STATUSES:
                yield f"event: done\ndata: {json.dumps({'run_id': run_id, 'status': record['status']})}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/runs/{run_id}", response_model=AsyncRunStatusResponse)
def cancel_async_run(
    run_id: str,
    user: AuthUser = Depends(current_user),
) -> AsyncRunStatusResponse:
    try:
        record = RUN_MANAGER.cancel(run_id, user.user_id, is_admin=user.is_admin)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="运行记录不存在。") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="无权取消该运行。") from exc
    log_audit_event(
        "run.cancel",
        actor_id=user.user_id,
        role=user.role,
        resource_type="run",
        resource_id=run_id,
        outcome=record["status"],
    )
    return AsyncRunStatusResponse(**record)


@app.post("/api/runs/{run_id}/review", response_model=AsyncRunStatusResponse, status_code=202)
def review_async_run(
    run_id: str,
    decision: ReviewDecisionRequest,
    user: AuthUser = Depends(admin_user),
) -> AsyncRunStatusResponse:
    try:
        RUN_MANAGER.get_for_owner(run_id, user.user_id, is_admin=True)
        record = RUN_MANAGER.resume(
            run_id,
            user.user_id,
            lambda resumed_run_id, _: _resume_async_request(resumed_run_id, decision.approved, user),
            timeout_seconds=300.0,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="运行记录不存在。") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AsyncRunStatusResponse(**record)


@app.post("/api/run-jobpilot", response_model=RunJobPilotResponse)
def run_jobpilot_api(
    payload: RunJobPilotRequest,
    request: Request,
    user: AuthUser = Depends(current_user),
) -> RunJobPilotResponse:
    enforce_run_rate_limit(request, user)
    run_id, run_dir = _make_run_dir()
    api_available = _api_available()

    try:
        initial_state = _build_initial_state(payload, run_dir, api_available)
        initial_state["run_id"] = run_id
        initial_state["_deadline_epoch"] = datetime.now(timezone.utc).timestamp() + payload.timeout_seconds
        final_state = run_jobpilot(
            initial_state,
            thread_id=run_id,
            checkpoint_path=CHECKPOINT_PATH,
        )
        output_paths = write_outputs(final_state, payload.target_role)
        final_report_markdown = build_final_report_markdown(final_state, payload.target_role)
    except Exception as exc:
        log_audit_event(
            "run.sync",
            actor_id=user.user_id,
            role=user.role,
            resource_type="run",
            resource_id=run_id,
            outcome="failed",
            metadata={"error_type": type(exc).__name__},
        )
        raise HTTPException(status_code=500, detail=f"JobPilot 运行失败：{exc}") from exc

    log_audit_event(
        "run.sync",
        actor_id=user.user_id,
        role=user.role,
        resource_type="run",
        resource_id=run_id,
        outcome=str(final_state.get("workflow_status") or "completed"),
    )
    return RunJobPilotResponse(
        run_id=run_id,
        api_available=api_available,
        matched_jobs=_as_list(final_state.get("matched_jobs")),
        gaps=_as_list(final_state.get("gaps")),
        resume_suggestions=_as_list(final_state.get("resume_suggestions")),
        trace=_as_list(final_state.get("trace")),
        token_usage=final_state.get("token_usage", {}),
        final_report=final_state.get("final_report", {}),
        final_report_markdown=final_report_markdown,
        output_paths={key: str(value) for key, value in output_paths.items()},
        workflow_status=str(final_state.get("workflow_status") or "completed"),
        review_required=bool(final_state.get("review_required")),
        review_reason=final_state.get("review_reason"),
        checkpoint_backend=final_state.get("checkpoint_backend"),
    )


@app.post("/api/record-jobs", response_model=RecordJobsResponse)
def record_jobs_api(
    request: RecordJobsRequest,
    user: AuthUser = Depends(admin_user),
) -> RecordJobsResponse:
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
            records_path=DEFAULT_RECORDS_PATH,
            jd_dir=DEFAULT_JD_FOLDER,
        )
        job_file_by_id = {job.get("job_id"): job for job in jobs}
        for job in db_jobs:
            file_job = job_file_by_id.get(job.get("job_id"))
            if file_job:
                job["record_path"] = file_job.get("record_path")
                job["jd_file_path"] = file_job.get("jd_file_path")
        index_refreshed, index_warning = _refresh_job_retrieval_store()
    except Exception as exc:
        log_audit_event(
            "job.upsert",
            actor_id=user.user_id,
            role=user.role,
            resource_type="job",
            outcome="failed",
            metadata={"count": len(jd_texts), "error_type": type(exc).__name__},
        )
        raise HTTPException(status_code=500, detail=f"保存岗位失败：{exc}") from exc

    log_audit_event(
        "job.upsert",
        actor_id=user.user_id,
        role=user.role,
        resource_type="job",
        outcome="success",
        metadata={"count": len(db_jobs), "index_refreshed": index_refreshed},
    )
    return RecordJobsResponse(
        saved_count=len(db_jobs),
        jobs=db_jobs,
        record_path=str(DEFAULT_RECORDS_PATH),
        jd_folder=str(DEFAULT_JD_FOLDER),
        db_path=str(DEFAULT_JOB_DB_PATH),
        vector_store_dir=str(_default_vector_store_dir()),
        index_refreshed=index_refreshed,
        index_warning=index_warning,
    )


@app.get("/api/jobs", response_model=JobsListResponse)
def list_jobs_api(
    include_inactive: bool = False,
    user: AuthUser = Depends(current_user),
) -> JobsListResponse:
    if include_inactive and not user.is_admin:
        raise HTTPException(status_code=403, detail="只有管理员可以查看已停用岗位。")
    try:
        jobs = list_jobs(db_path=DEFAULT_JOB_DB_PATH, include_inactive=include_inactive)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取岗位库失败：{exc}") from exc
    return JobsListResponse(jobs=jobs, count=len(jobs), db_path=str(DEFAULT_JOB_DB_PATH))


@app.delete("/api/jobs/{job_id}", response_model=DeleteJobResponse)
def delete_job_api(
    job_id: str,
    user: AuthUser = Depends(admin_user),
) -> DeleteJobResponse:
    deleted = soft_delete_job(job_id, db_path=DEFAULT_JOB_DB_PATH)
    if not deleted:
        raise HTTPException(status_code=404, detail="岗位不存在或已经停用。")
    index_refreshed, index_warning = _refresh_job_retrieval_store()
    log_audit_event(
        "job.deactivate",
        actor_id=user.user_id,
        role=user.role,
        resource_type="job",
        resource_id=job_id,
        metadata={"index_refreshed": index_refreshed},
    )
    return DeleteJobResponse(
        deleted=True,
        job_id=job_id,
        index_refreshed=index_refreshed,
        index_warning=index_warning,
    )


@app.post("/api/jobs/{job_id}/restore", response_model=RestoreJobResponse)
def restore_job_api(
    job_id: str,
    user: AuthUser = Depends(admin_user),
) -> RestoreJobResponse:
    restored = restore_job(job_id, db_path=DEFAULT_JOB_DB_PATH)
    if not restored:
        raise HTTPException(status_code=404, detail="岗位不存在或已经启用。")
    index_refreshed, index_warning = _refresh_job_retrieval_store()
    log_audit_event(
        "job.restore",
        actor_id=user.user_id,
        role=user.role,
        resource_type="job",
        resource_id=job_id,
        metadata={"index_refreshed": index_refreshed},
    )
    return RestoreJobResponse(
        restored=True,
        job_id=job_id,
        index_refreshed=index_refreshed,
        index_warning=index_warning,
    )


@app.get("/api/latest-trace")
def latest_trace(_: AuthUser = Depends(admin_user)) -> dict[str, Any]:
    if not LATEST_TRACE_PATH.exists():
        raise HTTPException(status_code=404, detail="尚未生成最新 trace 文件。")
    try:
        return {"trace": json.loads(LATEST_TRACE_PATH.read_text(encoding="utf-8"))}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"最新 trace JSON 格式无效：{exc}") from exc


@app.get("/api/latest-report")
def latest_report(_: AuthUser = Depends(admin_user)) -> dict[str, Any]:
    if not LATEST_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="尚未生成最新 final_report 文件。")
    return {"markdown": LATEST_REPORT_PATH.read_text(encoding="utf-8")}


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
