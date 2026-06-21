from pathlib import Path

from fastapi.testclient import TestClient

from src import api as api_module


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(api_module, "_api_available", lambda: False)

    client = TestClient(api_module.app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["api_available"] is False


def test_run_jobpilot_endpoint_builds_state(tmp_path, monkeypatch):
    captured_state = {}

    def fake_run_jobpilot(initial_state):
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
    assert captured_state["target_role"] == "AI Agent Intern"
    assert captured_state["retrieval_top_k"] == 5
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

    def fake_run_jobpilot(initial_state):
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
    jd_folder = Path(captured_state["jd_folder"])
    assert jd_folder.exists()
    assert len(list(jd_folder.glob("*.txt"))) == 1


def test_record_jobs_endpoint_saves_jsonl_and_jd_files(tmp_path, monkeypatch):
    records_path = tmp_path / "jobs_csv" / "job_records.jsonl"
    jd_folder = tmp_path / "sample_jds"
    db_path = tmp_path / "jobpilot.db"

    monkeypatch.setattr(api_module, "DEFAULT_RECORDS_PATH", records_path)
    monkeypatch.setattr(api_module, "DEFAULT_JD_FOLDER", jd_folder)
    monkeypatch.setattr(api_module, "DEFAULT_JOB_DB_PATH", db_path)
    monkeypatch.setattr(api_module, "ROOT_DIR", tmp_path)

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
    assert Path(data["db_path"]).exists()
    assert records_path.exists()
    assert Path(data["jobs"][0]["jd_file_path"]).exists()


def test_jobs_list_and_delete_endpoints(tmp_path, monkeypatch):
    db_path = tmp_path / "jobpilot.db"
    jd_folder = tmp_path / "sample_jds"

    monkeypatch.setattr(api_module, "DEFAULT_JOB_DB_PATH", db_path)
    monkeypatch.setattr(api_module, "DEFAULT_JD_FOLDER", jd_folder)

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
