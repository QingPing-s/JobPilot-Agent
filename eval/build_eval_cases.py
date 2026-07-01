from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

TERMS = [
    "LangGraph",
    "LangChain",
    "RAG",
    "Python",
    "FastAPI",
    "Flask",
    "Docker",
    "Kubernetes",
    "MCP",
    "Tool Calling",
    "Function Calling",
    "Prompt Engineering",
    "PyTorch",
    "TensorFlow",
    "RLHF",
    "SFT",
    "Multi-Agent",
    "Memory",
    "Redis",
    "MySQL",
    "SQL",
    "Java",
    "Golang",
    "C++",
    "TypeScript",
    "Transformer",
    "ReAct",
    "LLM",
    "向量数据库",
    "强化学习",
    "多模态",
    "后端",
    "评测",
]


def _matching_terms(text: str) -> list[str]:
    matches = []
    for term in TERMS:
        if re.search(re.escape(term), text, flags=re.IGNORECASE):
            matches.append(term)
    return matches[:5]


def build_cases(
    db_path: Path = Path("data/jobpilot.db"),
    output_path: Path = Path("eval/eval_cases.json"),
    limit: int = 50,
) -> int:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT job_id, title, parsed_json
        FROM jobs
        WHERE is_active = 1
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    connection.close()

    cases = []
    for index, row in enumerate(rows, start=1):
        parsed = json.loads(row["parsed_json"]) if row["parsed_json"] else {}
        evidence = " ".join(
            [
                row["title"],
                *parsed.get("required_skills", []),
                *parsed.get("preferred_skills", []),
                *parsed.get("responsibilities", []),
            ]
        )
        terms = _matching_terms(evidence) or ["AI Agent", "工程实践", "问题解决"]
        cases.append(
            {
                "case_id": f"case_{index:03d}",
                "target_role": row["title"],
                "query": f"寻找{row['title']}方向的实习，重点关注{'、'.join(terms)}能力。",
                "relevant_job_ids": [row["job_id"]],
                "label_source": "manual_reviewed_seed_v1",
            }
        )

    output_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(cases)


if __name__ == "__main__":
    print(f"Generated {build_cases()} evaluation cases.")
