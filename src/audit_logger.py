from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_AUDIT_PATH = Path(
    os.getenv("JOBPILOT_AUDIT_LOG", Path(os.getenv("JOBPILOT_DATA_DIR", "data")) / "audit.jsonl")
)
_LOCK = threading.Lock()


def log_audit_event(
    action: str,
    *,
    actor_id: str,
    role: str,
    resource_type: str,
    resource_id: str | None = None,
    outcome: str = "success",
    metadata: dict[str, Any] | None = None,
    path: str | Path = DEFAULT_AUDIT_PATH,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor_id": actor_id,
        "role": role,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "outcome": outcome,
        "metadata": metadata or {},
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
