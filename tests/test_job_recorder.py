import json
from pathlib import Path

from src.job_recorder import extract_job_record, save_job_record, save_job_records


def test_extract_job_record_from_structured_text():
    record = extract_job_record(
        """
Title: AI Agent Intern
Company: Example AI
Location: 北京
Salary: 300-500元/天
Responsibilities:
- Build Agent workflows
Requirements:
- Python
- RAG
Preferred:
- LangGraph
""",
        filename="agent.txt",
        source="test",
    )

    assert record["title"] == "AI Agent Intern"
    assert record["company"] == "Example AI"
    assert record["location"] == "北京"
    assert record["required_skills"] == ["Python", "RAG"]
    assert record["preferred_skills"] == ["LangGraph"]
    assert record["job_id"].startswith("job_agent_example_ai_ai_agent_intern_")


def test_save_job_record_writes_jsonl_and_jd_file(tmp_path):
    records_path = tmp_path / "jobs_csv" / "job_records.jsonl"
    jd_dir = tmp_path / "sample_jds"
    raw_text = "Title: RAG Intern\nCompany: Search AI\nRequirements:\n- Python\n- ChromaDB"

    first = save_job_record(raw_text, records_path=records_path, jd_dir=jd_dir)
    second = save_job_record(raw_text, records_path=records_path, jd_dir=jd_dir)

    lines = records_path.read_text(encoding="utf-8").splitlines()
    saved = json.loads(lines[0])

    assert len(lines) == 1
    assert first["already_exists"] is False
    assert second["already_exists"] is True
    assert saved["title"] == "RAG Intern"
    assert Path(first["jd_file_path"]).exists()
    assert "Raw Text:" in Path(first["jd_file_path"]).read_text(encoding="utf-8")


def test_save_job_records_skips_empty_text(tmp_path):
    records_path = tmp_path / "jobs_csv" / "job_records.jsonl"
    jd_dir = tmp_path / "sample_jds"

    saved = save_job_records(
        ["", "Title: LLM Intern\nCompany: Example AI"],
        jd_filenames=["empty.txt", "llm.txt"],
        records_path=records_path,
        jd_dir=jd_dir,
    )

    assert len(saved) == 1
    assert saved[0]["source_filename"] == "llm.txt"
