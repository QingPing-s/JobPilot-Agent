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
    organized_text = Path(first["jd_file_path"]).read_text(encoding="utf-8")
    assert "Required Skills:" in organized_text
    assert "Raw Text:" not in organized_text


def test_extract_job_record_organizes_unstructured_chinese_jd():
    record = extract_job_record(
        """
AI Agent开发实习生 300-500元/天
北京 5天/周 6个月 本科
岗位职责：
1. 参与 Agent 工作流和工具调用模块开发
2. 建设 RAG 知识库
任职要求：
1. 熟练使用 Python
2. 了解 LangGraph
加分项：
- 有 ChromaDB 项目经验
示例科技 · 招聘者
""",
        filename="agent.txt",
    )

    assert record["title"] == "AI Agent开发实习生"
    assert record["company"] == "示例科技"
    assert record["location"] == "北京"
    assert record["salary"] == "300-500元/天"
    assert record["duration"] == "6个月"
    assert record["education"] == "本科"
    assert record["responsibilities"] == ["参与 Agent 工作流和工具调用模块开发", "建设 RAG 知识库"]
    assert record["required_skills"] == ["熟练使用 Python", "了解 LangGraph"]
    assert record["preferred_skills"] == ["有 ChromaDB 项目经验"]
    assert "Title: AI Agent开发实习生" in record["organized_text"]
    assert "Raw Text:" not in record["organized_text"]

    organized_record = extract_job_record(record["organized_text"], filename="agent.txt")
    assert organized_record["job_id"] == record["job_id"]


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


def test_extract_job_record_supports_numbered_recruitment_template():
    record = extract_job_record(
        """
一、基础招聘信息
1. 岗位名称：解决方案实习生
2. 薪资：200-300元/天
3. 工作地点：北京海淀区学院路
4. 出勤要求：每周4天，实习期3个月及以上
5. 学历要求：本科及以上在读
二、公司信息
公司全称：北京示例科技有限责任公司
四、工作内容
1. 开发垂直场景 AI 应用 Demo
2. 搭建 RAG 与 Agent 流程
五、任职要求
1. 掌握 Python
2. 能调用大模型 API
六、加分项目
1. 有大模型项目经验
""",
        filename="solution_intern.txt",
    )

    assert record["title"] == "解决方案实习生"
    assert record["company"] == "北京示例科技有限责任公司"
    assert record["location"] == "北京海淀区学院路"
    assert record["salary"] == "200-300元/天"
    assert record["duration"] == "3个月及以上"
    assert record["education"] == "本科及以上在读"
    assert record["responsibilities"] == ["开发垂直场景 AI 应用 Demo", "搭建 RAG 与 Agent 流程"]
    assert record["required_skills"] == ["掌握 Python", "能调用大模型 API"]
    assert record["preferred_skills"] == ["有大模型项目经验"]
