import json

from src.trace_logger import TraceLogger


def test_trace_logger_records_start_end_error_and_saves_pretty_json(tmp_path):
    logger = TraceLogger()

    logger.log_node_start("profile_node", {"input_count": 1, "api_key": "secret"})
    logger.log_node_end("profile_node", {"output_count": 1, "raw_text": "x" * 300})
    logger.log_error("jd_parse_node", "failed")

    path = tmp_path / "latest_trace.json"
    logger.save(str(path))

    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    assert text.startswith("[\n  {")
    assert [item["event_type"] for item in data] == ["start", "end", "error"]
    assert data[0]["payload"]["api_key"] == "[REDACTED]"
    assert data[1]["payload"]["raw_text"].endswith("...")
    assert data[2]["payload"]["error"] == "failed"
