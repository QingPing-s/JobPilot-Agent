import argparse
import json

from src import main as main_module


def _final_state():
    return {
        "candidate_profile": {
            "name": "Alex",
            "target_roles": ["AI Agent Intern"],
        },
        "matched_jobs": [
            {
                "job_id": "job_1",
                "title": "AI Agent Intern",
                "company": "Example AI",
                "match_score": 91.5,
                "skill_overlap": ["Python", "LLM"],
                "missing_skills": ["LangGraph"],
                "recommendation": "Apply after adding LangGraph evidence.",
            }
        ],
        "gaps": [
            {
                "job_id": "job_1",
                "gaps": [
                    {
                        "type": "missing_skill",
                        "severity": "medium",
                        "description": "LangGraph evidence is missing.",
                        "suggestion": "Add a LangGraph project bullet.",
                    }
                ],
            }
        ],
        "resume_suggestions": [
            {
                "job_id": "job_1",
                "suggestions": [
                    {
                        "section": "Projects",
                        "original_problem": "Project evidence is too generic.",
                        "suggestion": "Tie the project to agent workflow requirements.",
                        "improved_example": "Built a Python LLM agent prototype with trace logging.",
                    }
                ],
            }
        ],
        "trace": [
            {
                "node": "profile_node",
                "status": "success",
                "message": "ok",
            }
        ],
    }


def test_validate_env_returns_false_without_api_key(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert main_module._validate_env() is False
    assert "OPENAI_API_KEY 未配置" in capsys.readouterr().out


def test_write_outputs_creates_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(main_module, "TRACE_DIR", tmp_path / "traces")

    state = _final_state()
    paths = main_module.write_outputs(state, "AI Agent Intern")

    assert paths["matched_jobs"].exists()
    assert paths["resume_suggestions"].exists()
    assert paths["final_report"].exists()
    assert paths["trace"].exists()
    assert json.loads(paths["matched_jobs"].read_text(encoding="utf-8"))[0]["job_id"] == "job_1"
    report_text = paths["final_report"].read_text(encoding="utf-8")
    assert "Top 5 推荐岗位" in report_text
    assert "Token 消耗" not in report_text
    assert "求职信草稿" not in report_text
    assert state["final_report"]["target_role"] == "AI Agent Intern"


def test_run_calls_run_jobpilot_and_prints_paths(tmp_path, monkeypatch, capsys):
    profile_path = tmp_path / "user_profile.json"
    jd_folder = tmp_path / "jds"
    profile_path.write_text('{"name": "Alex"}', encoding="utf-8")
    jd_folder.mkdir()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(main_module, "TRACE_DIR", tmp_path / "traces")

    from src import graph as graph_module

    captured_state = {}

    def fake_run_jobpilot(initial_state):
        captured_state.update(initial_state)
        return _final_state()

    monkeypatch.setattr(graph_module, "run_jobpilot", fake_run_jobpilot)

    args = argparse.Namespace(
        profile=str(profile_path),
        jd_folder=str(jd_folder),
        target_role="AI Agent Intern",
    )
    result = main_module.run(args)

    assert captured_state["user_profile_path"] == str(profile_path)
    assert captured_state["jd_folder"] == str(jd_folder)
    assert captured_state["target_role"] == "AI Agent Intern"
    assert captured_state["api_available"] is True
    assert result["matched_jobs"][0]["job_id"] == "job_1"
    assert "岗位匹配结果:" in capsys.readouterr().out
