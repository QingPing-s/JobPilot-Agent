import json

import pytest

from src.tools import (
    generate_job_id,
    list_jd_files,
    load_jd_files,
    load_json,
    load_text_file,
    load_user_profile,
    normalize_skill,
    save_json,
    save_markdown,
    simple_skill_overlap,
)


def test_load_json(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text('{"name": "Alex"}', encoding="utf-8")
    assert load_json(path)["name"] == "Alex"


def test_load_text_file(tmp_path):
    path = tmp_path / "jd.txt"
    path.write_text("Title: AI Agent Intern", encoding="utf-8")
    assert "AI Agent" in load_text_file(path)


def test_list_jd_files(tmp_path):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.md").write_text("B", encoding="utf-8")
    assert [path.name for path in list_jd_files(tmp_path)] == ["a.txt"]


def test_load_user_profile(tmp_path):
    path = tmp_path / "user_profile.json"
    path.write_text('{"name": "Alex", "skills": ["Python"]}', encoding="utf-8")

    profile = load_user_profile(str(path))

    assert profile["name"] == "Alex"
    assert profile["skills"] == ["Python"]


def test_load_jd_files(tmp_path):
    (tmp_path / "agent.txt").write_text("Agent JD", encoding="utf-8")
    (tmp_path / "rag.txt").write_text("RAG JD", encoding="utf-8")
    (tmp_path / "notes.md").write_text("Ignored", encoding="utf-8")

    jobs = load_jd_files(str(tmp_path))

    assert jobs == [
        {"filename": "agent.txt", "raw_text": "Agent JD"},
        {"filename": "rag.txt", "raw_text": "RAG JD"},
    ]


def test_save_json_keeps_chinese(tmp_path):
    path = tmp_path / "outputs" / "result.json"
    data = {"message": "中文内容", "skills": ["Python"]}

    save_json(data, str(path))

    saved_text = path.read_text(encoding="utf-8")
    assert "中文内容" in saved_text
    assert json.loads(saved_text) == data


def test_save_markdown(tmp_path):
    path = tmp_path / "outputs" / "report.md"

    save_markdown("# 报告\n\n内容", str(path))

    assert path.read_text(encoding="utf-8") == "# 报告\n\n内容"


def test_generate_job_id():
    assert generate_job_id("agent_intern_01.txt", 1) == "job_agent_intern_01"
    assert generate_job_id("RAG Intern.txt", 2) == "job_rag_intern_02"


@pytest.mark.parametrize(
    ("skill", "expected"),
    [
        (" LLM ", "large language model"),
        ("rag", "retrieval augmented generation"),
        ("Py", "python"),
        ("Machine   Learning", "machine learning"),
    ],
)
def test_normalize_skill(skill, expected):
    assert normalize_skill(skill) == expected


def test_simple_skill_overlap():
    overlap = simple_skill_overlap(
        ["Py", "RAG", "SQL", "RAG"],
        ["Python", "retrieval augmented generation", "Docker"],
    )

    assert overlap == ["python", "retrieval augmented generation"]


def test_load_json_invalid_file_raises_clear_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON 文件格式无效"):
        load_json(path)


def test_save_json_rejects_invalid_data_type(tmp_path):
    with pytest.raises(TypeError, match="dict 或 list"):
        save_json("not-json-container", str(tmp_path / "bad.json"))
