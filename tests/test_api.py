import time
from pathlib import Path

from fastapi.testclient import TestClient

from src import api as api_module
from src.document_parser import DocumentExtraction
from src.ocr_service import OCRExtraction, OCRServiceError
from src.run_manager import RunManager, RunStore


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(api_module, "_api_available", lambda: False)

    client = TestClient(api_module.app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["api_available"] is False


def test_profile_ocr_endpoint_returns_candidate_profile(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "extract_image_text",
        lambda image_bytes, filename: OCRExtraction(
            text="AAA建材\n人工智能专业\nPython RAG",
            lines=["AAA建材", "人工智能专业", "Python RAG"],
            scores=[0.98, 0.95, 0.9],
            average_confidence=0.9433,
        ),
    )
    monkeypatch.setattr(api_module, "_api_available", lambda: False)

    client = TestClient(api_module.app)
    response = client.post(
        "/api/profile/ocr?filename=resume.png&target_role=AI%20Agent%20Intern",
        content=b"image-bytes",
        headers={"Content-Type": "image/png"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "resume.png"
    assert data["line_count"] == 3
    assert data["candidate_profile"]["skills"] == ["Python", "RAG"]
    assert data["profile_extraction_mode"] == "rule_based"
    assert data["warnings"]


def test_profile_ocr_endpoint_rejects_invalid_image(monkeypatch):
    def fail_ocr(image_bytes, filename):
        raise OCRServiceError("图片无效。")

    monkeypatch.setattr(api_module, "extract_image_text", fail_ocr)
    client = TestClient(api_module.app)
    response = client.post(
        "/api/profile/ocr?filename=resume.png",
        content=b"invalid",
        headers={"Content-Type": "image/png"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "图片无效。"


def test_profile_document_endpoint_returns_candidate_profile(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "extract_document_text",
        lambda document_bytes, filename: DocumentExtraction(
            text="AAA建材\n人工智能专业\nPython RAG",
            extraction_method="docx",
            line_count=3,
        ),
    )
    monkeypatch.setattr(api_module, "_api_available", lambda: False)

    client = TestClient(api_module.app)
    response = client.post(
        "/api/profile/document?filename=resume.docx&target_role=AI%20Agent%20Intern",
        content=b"document-bytes",
        headers={"Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "resume.docx"
    assert data["extraction_method"] == "docx"
    assert data["candidate_profile"]["skills"] == ["Python", "RAG"]
    assert data["profile_extraction_mode"] == "rule_based"


def test_profile_document_endpoint_accepts_structured_json():
    client = TestClient(api_module.app)
    response = client.post(
        "/api/profile/document?filename=resume.json",
        content=b"""
        {
          "name": "AAA",
          "education": [],
          "skills": ["Python"],
          "soft_skills": [],
          "projects": [],
          "internships": [],
          "target_roles": [],
          "preferences": {}
        }
        """,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["candidate_profile"]["name"] == "AAA"
    assert data["extraction_method"] == "json"
    assert data["warnings"] == []


def test_staged_job_database_hydrates_and_persists(tmp_path, monkeypatch):
    working_path = tmp_path / "working" / "jobpilot.db"
    persistent_path = tmp_path / "persistent" / "jobpilot.db"
    persistent_path.parent.mkdir(parents=True)
    persistent_path.write_bytes(b"original")
    monkeypatch.setenv("JOBPILOT_JOB_DB_PATH", str(working_path))
    monkeypatch.setattr(api_module, "DEFAULT_JOB_DB_PATH", working_path)
    monkeypatch.setattr(api_module, "PERSISTENT_JOB_DB_PATH", persistent_path)

    api_module._hydrate_job_database()
    assert working_path.read_bytes() == b"original"

    working_path.write_bytes(b"updated")
    api_module._persist_job_database()
    assert persistent_path.read_bytes() == b"updated"


def test_public_access_allows_job_reads_but_rejects_job_writes(monkeypatch):
    monkeypatch.setenv("JOBPILOT_AUTH_ENABLED", "true")
    monkeypatch.setenv("JOBPILOT_PUBLIC_ACCESS", "true")

    client = TestClient(api_module.app)

    assert client.get("/api/jobs").status_code == 200
    denied = client.post("/api/record-jobs", json={"jd_texts": ["Title: Agent Intern"]})
    assert denied.status_code == 403
    assert client.get("/api/health").json()["public_access"] is True


def test_run_jobpilot_endpoint_builds_state(tmp_path, monkeypatch):
    captured_state = {}

    def fake_run_jobpilot(initial_state, **kwargs):
        captured_state.update(initial_state)
        return {
            "matched_jobs": [
                {
                    "job_id": "job_agent_01",
                    "title": "AI Agent Intern",
                    "company": "Example AI",
                    "match_score": 88.0,
                    "skill_overlap": ["Python"],
                    "missing_skills": [],
                    "matched_projects": [],
                    "reason": "Good fit.",
                    "recommendation": "Apply.",
                }
            ],
            "gaps": [],
            "resume_suggestions": [],
            "trace": [{"node": "profile_node", "status": "success"}],
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "calls": 1},
        }

    def fake_write_outputs(state, target_role):
        state["final_report"] = {"target_role": target_role}
        return {
            "matched_jobs": Path("outputs/matched_jobs.json"),
            "resume_suggestions": Path("outputs/resume_suggestions.json"),
            "final_report": Path("outputs/final_report.md"),
            "trace": Path("traces/latest_trace.json"),
        }

    monkeypatch.setattr(api_module, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(api_module, "_api_available", lambda: False)
    monkeypatch.setattr(api_module, "run_jobpilot", fake_run_jobpilot)
    monkeypatch.setattr(api_module, "write_outputs", fake_write_outputs)

    client = TestClient(api_module.app)
    response = client.post(
        "/api/run-jobpilot",
        json={
            "target_role": "AI Agent Intern",
            "user_profile_json": {
                "name": "Alex",
                "education": [],
                "skills": ["Python"],
                "projects": [],
                "internships": [],
                "target_roles": ["AI Agent Intern"],
                "preferences": {},
            },
            "jd_texts": ["Title: AI Agent Intern\nRequirements:\n- Python"],
            "jd_filenames": ["agent.txt"],
            "retrieval_top_k": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["api_available"] is False
    assert data["matched_jobs"][0]["job_id"] == "job_agent_01"
    assert data["token_usage"]["total_tokens"] == 14
    assert data["token_usage"]["calls"] == 1
    assert captured_state["target_role"] == "AI Agent Intern"
    assert captured_state["retrieval_top_k"] == 5
    assert captured_state["rerank_top_k"] == 10
    assert captured_state["llm_rerank_top_n"] == 5
    assert captured_state["llm_match_top_n"] == 3
    assert captured_state["gap_top_n"] == 1
    assert captured_state["resume_top_n"] == 1
    assert captured_state["use_llm_deep_analysis"] is False
    assert captured_state["api_available"] is False
    assert Path(captured_state["user_profile_path"]).exists()
    assert Path(captured_state["jd_folder"]).exists()


def test_run_jobpilot_can_use_sqlite_job_library(tmp_path, monkeypatch):
    captured_state = {}
    db_path = tmp_path / "jobpilot.db"

    from src.job_store import upsert_job

    upsert_job(
        raw_text="Title: Agent Intern\nCompany: Example AI\nRequirements:\n- Python",
        filename="agent.txt",
        source="test",
        db_path=db_path,
    )

    def fake_run_jobpilot(initial_state, **kwargs):
        captured_state.update(initial_state)
        return {"matched_jobs": [], "gaps": [], "resume_suggestions": [], "trace": []}

    def fake_write_outputs(state, target_role):
        state["final_report"] = {"target_role": target_role}
        return {
            "matched_jobs": Path("outputs/matched_jobs.json"),
            "resume_suggestions": Path("outputs/resume_suggestions.json"),
            "final_report": Path("outputs/final_report.md"),
            "trace": Path("traces/latest_trace.json"),
        }

    monkeypatch.setattr(api_module, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(api_module, "DEFAULT_JOB_DB_PATH", db_path)
    monkeypatch.setattr(api_module, "_api_available", lambda: False)
    monkeypatch.setattr(api_module, "run_jobpilot", fake_run_jobpilot)
    monkeypatch.setattr(api_module, "write_outputs", fake_write_outputs)
    monkeypatch.setattr(api_module, "_default_vector_store_dir", lambda: tmp_path / "vector_store")

    client = TestClient(api_module.app)
    response = client.post(
        "/api/run-jobpilot",
        json={
            "target_role": "AI Agent Intern",
            "user_profile_json": {
                "name": "Alex",
                "education": [],
                "skills": ["Python"],
                "projects": [],
                "internships": [],
                "target_roles": ["AI Agent Intern"],
                "preferences": {},
            },
            "use_job_library": True,
        },
    )

    assert response.status_code == 200
    assert captured_state["jd_folder"] == ""
    assert captured_state["skip_jd_parse"] is True
    assert captured_state["job_source"] == "sqlite_job_library"
    assert captured_state["vector_store_dir"] == str(tmp_path / "vector_store")
    assert len(captured_state["parsed_jobs"]) == 1
    assert captured_state["parsed_jobs"][0]["title"] == "Agent Intern"
    assert captured_state["parsed_jobs"][0]["required_skills"] == ["Python"]


def test_record_jobs_endpoint_saves_jsonl_and_jd_files(tmp_path, monkeypatch):
    records_path = tmp_path / "jobs_csv" / "job_records.jsonl"
    jd_folder = tmp_path / "sample_jds"
    db_path = tmp_path / "jobpilot.db"

    monkeypatch.setattr(api_module, "DEFAULT_RECORDS_PATH", records_path)
    monkeypatch.setattr(api_module, "DEFAULT_JD_FOLDER", jd_folder)
    monkeypatch.setattr(api_module, "DEFAULT_JOB_DB_PATH", db_path)
    monkeypatch.setattr(api_module, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(api_module, "_refresh_job_retrieval_store", lambda: (True, None))

    client = TestClient(api_module.app)
    response = client.post(
        "/api/record-jobs",
        json={
            "source": "test",
            "jd_texts": [
                "Title: AI Agent Intern\nCompany: Example AI\nLocation: 北京\nRequirements:\n- Python\n- RAG"
            ],
            "jd_filenames": ["agent.txt"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["saved_count"] == 1
    assert data["jobs"][0]["title"] == "AI Agent Intern"
    assert data["jobs"][0]["has_parsed_job"] is True
    assert data["jobs"][0]["parsed_job"]["required_skills"] == ["Python", "RAG"]
    assert data["jobs"][0]["organized_text"].startswith("Title: AI Agent Intern")
    assert Path(data["db_path"]).exists()
    assert data["index_refreshed"] is True
    assert records_path.exists()
    assert Path(data["jobs"][0]["jd_file_path"]).exists()


def test_jobs_list_and_delete_endpoints(tmp_path, monkeypatch):
    db_path = tmp_path / "jobpilot.db"
    jd_folder = tmp_path / "sample_jds"

    monkeypatch.setattr(api_module, "DEFAULT_JOB_DB_PATH", db_path)
    monkeypatch.setattr(api_module, "DEFAULT_JD_FOLDER", jd_folder)
    monkeypatch.setattr(api_module, "_refresh_job_retrieval_store", lambda: (True, None))

    client = TestClient(api_module.app)
    save_response = client.post(
        "/api/record-jobs",
        json={
            "source": "test",
            "jd_texts": ["Title: Agent Intern\nCompany: Example AI\nRequirements:\n- Python"],
            "jd_filenames": ["agent.txt"],
        },
    )
    assert save_response.status_code == 200
    job_id = save_response.json()["jobs"][0]["job_id"]

    list_response = client.get("/api/jobs")
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1

    delete_response = client.delete(f"/api/jobs/{job_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True

    list_after_delete = client.get("/api/jobs")
    assert list_after_delete.status_code == 200
    assert list_after_delete.json()["count"] == 0


def test_latest_trace_and_report_endpoints(tmp_path, monkeypatch):
    trace_path = tmp_path / "latest_trace.json"
    report_path = tmp_path / "final_report.md"
    trace_path.write_text('[{"node": "profile_node", "status": "success"}]', encoding="utf-8")
    report_path.write_text("# Report", encoding="utf-8")

    monkeypatch.setattr(api_module, "LATEST_TRACE_PATH", trace_path)
    monkeypatch.setattr(api_module, "LATEST_REPORT_PATH", report_path)

    client = TestClient(api_module.app)

    trace_response = client.get("/api/latest-trace")
    report_response = client.get("/api/latest-report")

    assert trace_response.status_code == 200
    assert trace_response.json()["trace"][0]["node"] == "profile_node"
    assert report_response.status_code == 200
    assert report_response.json()["markdown"] == "# Report"


def test_async_run_lifecycle(tmp_path, monkeypatch):
    manager = RunManager(store=RunStore(tmp_path / "runs.db"), max_workers=1)
    monkeypatch.setattr(api_module, "RUN_MANAGER", manager)
    monkeypatch.setattr(
        api_module,
        "_execute_async_request",
        lambda payload, user, run_id, deadline: {
            "run_id": run_id,
            "workflow_status": "completed",
            "matched_jobs": [],
            "gaps": [],
            "resume_suggestions": [],
            "trace": [],
            "token_usage": {},
            "final_report": {},
            "final_report_markdown": "",
            "output_paths": {},
            "api_available": False,
        },
    )

    client = TestClient(api_module.app)
    created = client.post(
        "/api/runs",
        json={
            "target_role": "AI Agent Intern",
            "user_profile_json": {
                "name": "Alex",
                "education": [],
                "skills": ["Python"],
                "projects": [],
                "internships": [],
                "target_roles": [],
                "preferences": {},
            },
            "jd_texts": ["Title: Agent Intern\nRequirements:\n- Python"],
            "allow_cache": False,
        },
    )

    assert created.status_code == 202
    run_id = created.json()["run_id"]
    for _ in range(100):
        status = client.get(f"/api/runs/{run_id}")
        if status.json()["status"] == "completed":
            break
        time.sleep(0.01)
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["result"]["run_id"] == run_id


def test_authenticated_api_enforces_roles_and_run_ownership(tmp_path, monkeypatch):
    manager = RunManager(store=RunStore(tmp_path / "runs.db"), max_workers=1)
    monkeypatch.setattr(api_module, "RUN_MANAGER", manager)
    monkeypatch.setenv("JOBPILOT_AUTH_ENABLED", "true")
    monkeypatch.setenv("JOBPILOT_JWT_SECRET", "test-secret-with-at-least-thirty-two-bytes")
    monkeypatch.setenv("JOBPILOT_USER_USERNAME", "alice")
    monkeypatch.setenv("JOBPILOT_USER_PASSWORD", "alice-password")
    monkeypatch.setenv("JOBPILOT_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("JOBPILOT_ADMIN_PASSWORD", "admin-password")

    client = TestClient(api_module.app)
    user_token = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "alice-password"},
    ).json()["access_token"]
    admin_token = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin-password"},
    ).json()["access_token"]
    user_headers = {"Authorization": f"Bearer {user_token}"}
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    denied = client.post(
        "/api/record-jobs",
        headers=user_headers,
        json={"jd_texts": ["Title: Agent Intern"]},
    )
    assert denied.status_code == 403

    record = manager.store.create("private-run", "someone-else", "hash")
    assert record["owner_id"] == "someone-else"
    assert client.get("/api/runs/private-run", headers=user_headers).status_code == 403
    assert client.get("/api/runs/private-run", headers=admin_headers).status_code == 200


def test_request_hash_changes_with_job_library_version(tmp_path, monkeypatch):
    from src.job_store import upsert_job

    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(api_module, "DEFAULT_JOB_DB_PATH", db_path)
    request = api_module.RunJobPilotRequest(use_job_library=True)

    before = api_module._request_hash(request)
    upsert_job(
        raw_text="Title: Agent Intern\nCompany: Example AI",
        filename="agent.txt",
        source="test",
        db_path=db_path,
    )
    after = api_module._request_hash(request)

    assert before != after


def test_initialize_persistent_data_skips_refresh_when_vector_store_is_current(tmp_path, monkeypatch):
    db_path = tmp_path / "jobpilot.db"
    db_path.write_text("db", encoding="utf-8")
    vector_store_dir = tmp_path / "vector_store"
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    store_path = vector_store_dir / "job_documents.json"
    backend_path = vector_store_dir / "retriever_backend.json"
    store_path.write_text("{}", encoding="utf-8")
    backend_path.write_text("{}", encoding="utf-8")

    calls = {"refresh": 0}
    monkeypatch.setattr(api_module, "DEFAULT_JOB_DB_PATH", db_path)
    monkeypatch.setattr(api_module, "JOB_SEED_PATH", tmp_path / "missing_seed.json")
    monkeypatch.setattr(api_module, "_default_vector_store_dir", lambda: vector_store_dir)
    monkeypatch.setattr(api_module, "list_jobs", lambda **kwargs: [{"job_id": "job_001"}])
    monkeypatch.setattr(api_module, "list_parsed_jobs", lambda **kwargs: [{"job_id": "job_001"}])
    monkeypatch.setattr(api_module, "is_retrieval_store_current", lambda jobs, persist_dir: True)
    monkeypatch.setattr(api_module, "_refresh_job_retrieval_store", lambda: calls.__setitem__("refresh", calls["refresh"] + 1))

    api_module._initialize_persistent_data()

    assert calls["refresh"] == 0


def test_vector_store_refreshes_when_manifest_is_stale(tmp_path, monkeypatch):
    db_path = tmp_path / "jobpilot.db"
    db_path.write_text("db", encoding="utf-8")
    vector_store_dir = tmp_path / "vector_store"
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    (vector_store_dir / "job_documents.json").write_text("{}", encoding="utf-8")
    (vector_store_dir / "retriever_backend.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(api_module, "DEFAULT_JOB_DB_PATH", db_path)
    monkeypatch.setattr(api_module, "_default_vector_store_dir", lambda: vector_store_dir)
    monkeypatch.setattr(api_module, "list_parsed_jobs", lambda **kwargs: [{"job_id": "job_001"}])
    monkeypatch.setattr(api_module, "is_retrieval_store_current", lambda jobs, persist_dir: False)

    assert api_module._vector_store_needs_refresh() is True


def test_refresh_job_retrieval_store_reports_vector_fallback(tmp_path, monkeypatch):
    def fake_build_chroma_store(jobs, persist_dir):
        fake_build_chroma_store.last_stats = {
            "backend": "simple",
            "warning": "embedding model unavailable",
        }

    monkeypatch.setattr(api_module, "list_parsed_jobs", lambda **kwargs: [{"job_id": "job_001"}])
    monkeypatch.setattr(api_module, "build_chroma_store", fake_build_chroma_store)
    monkeypatch.setattr(api_module, "_default_vector_store_dir", lambda: tmp_path / "vector_store")

    refreshed, warning = api_module._refresh_job_retrieval_store()

    assert refreshed is True
    assert warning == "embedding model unavailable"


def test_initialize_persistent_data_refreshes_when_vector_store_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "jobpilot.db"
    db_path.write_text("db", encoding="utf-8")
    vector_store_dir = tmp_path / "vector_store"

    calls = {"refresh": 0}

    def mark_refresh():
        calls["refresh"] += 1
        return True, None

    monkeypatch.setattr(api_module, "DEFAULT_JOB_DB_PATH", db_path)
    monkeypatch.setattr(api_module, "JOB_SEED_PATH", tmp_path / "missing_seed.json")
    monkeypatch.setattr(api_module, "_default_vector_store_dir", lambda: vector_store_dir)
    monkeypatch.setattr(api_module, "list_jobs", lambda **kwargs: [{"job_id": "job_001"}])
    monkeypatch.setattr(api_module, "_refresh_job_retrieval_store", mark_refresh)

    api_module._initialize_persistent_data()

    assert calls["refresh"] == 1
