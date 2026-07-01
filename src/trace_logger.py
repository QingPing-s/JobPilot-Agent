from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {
    "api_key",
    "openai_api_key",
    "authorization",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "token",
}
LONG_TEXT_KEYS = {
    "raw_text",
    "user_profile_text",
    "jd_text",
    "content",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, max_length: int = 200) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def _safe_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        safe = {}
        for key, value in payload.items():
            normalized_key = str(key).lower()
            is_credential = (
                normalized_key in SENSITIVE_KEYS
                or normalized_key.endswith("_api_key")
                or normalized_key.endswith("_password")
                or normalized_key.endswith("_secret")
            )
            if is_credential:
                safe[key] = "[REDACTED]"
            elif normalized_key in LONG_TEXT_KEYS:
                safe[key] = _truncate(str(value))
            else:
                safe[key] = _safe_payload(value)
        return safe
    if isinstance(payload, list):
        return [_safe_payload(item) for item in payload[:20]]
    if isinstance(payload, str):
        return _truncate(payload)
    return payload


class TraceLogger:
    """Collect and save readable Agent execution traces."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.records: list[dict[str, Any]] = []

    def _append(self, node_name: str, event_type: str, payload: dict) -> None:
        self.records.append(
            {
                "timestamp": utc_timestamp(),
                "node": node_name,
                "event_type": event_type,
                "payload": _safe_payload(payload),
            }
        )

    def log_node_start(self, node_name: str, input_summary: dict) -> None:
        self._append(node_name, "start", input_summary)

    def log_node_end(self, node_name: str, output_summary: dict) -> None:
        self._append(node_name, "end", output_summary)

    def log_error(self, node_name: str, error: str) -> None:
        self._append(node_name, "error", {"error": error})

    def log(self, event: str, **payload: Any) -> None:
        """Backward-compatible generic log method."""
        self._append(event, "end", payload)
        if self.path is not None:
            self.save(str(self.path))

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
