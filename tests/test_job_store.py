import sqlite3

from src.job_store import list_jobs, list_parsed_jobs, restore_job, soft_delete_job, upsert_job


def test_upsert_job_stores_parsed_job_cache(tmp_path):
    db_path = tmp_path / "jobpilot.db"

    saved = upsert_job(
        raw_text=(
            "Title: AI Agent Intern\n"
            "Company: Example AI\n"
            "Location: Beijing\n"
            "Requirements:\n"
            "- Python\n"
            "- RAG\n"
            "Preferred:\n"
            "- LangGraph\n"
        ),
        filename="agent.txt",
        source="test",
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT parsed_json, content_hash FROM jobs WHERE job_id = ?", (saved["job_id"],)).fetchone()

    assert row is not None
    assert row[0]
    assert row[1]
    assert saved["has_parsed_job"] is True

    jobs = list_jobs(db_path=db_path)
    parsed_jobs = list_parsed_jobs(db_path=db_path)

    assert jobs[0]["has_parsed_job"] is True
    assert parsed_jobs[0]["job_id"] == saved["job_id"]
    assert parsed_jobs[0]["required_skills"] == ["Python", "RAG"]
    assert parsed_jobs[0]["preferred_skills"] == ["LangGraph"]


def test_soft_delete_and_restore_job(tmp_path):
    db_path = tmp_path / "jobpilot.db"
    saved = upsert_job(
        raw_text="Title: Agent Intern\nCompany: Example AI",
        filename="agent.txt",
        source="test",
        db_path=db_path,
    )

    assert soft_delete_job(saved["job_id"], db_path=db_path) is True
    assert list_jobs(db_path=db_path) == []
    assert restore_job(saved["job_id"], db_path=db_path) is True
    assert list_jobs(db_path=db_path)[0]["job_id"] == saved["job_id"]
