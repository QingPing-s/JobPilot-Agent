from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import MatchResult


def _to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def export_match_result(result: MatchResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_dict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
