from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def export_seed(db_path: Path, output_path: Path) -> int:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT job_id, raw_text, source, source_filename
        FROM jobs
        WHERE is_active = 1
        ORDER BY job_id
        """
    ).fetchall()
    connection.close()

    records = [
        {
            "raw_text": row["raw_text"],
            "filename": row["source_filename"] or f"{row['job_id']}.txt",
            "source": row["source"] or "sanitized_seed",
        }
        for row in rows
        if str(row["raw_text"] or "").strip()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export active SQLite jobs as a reproducible seed file.")
    parser.add_argument("--db", type=Path, default=Path("data/jobpilot.db"))
    parser.add_argument("--output", type=Path, default=Path("data/job_seed.json"))
    args = parser.parse_args()
    count = export_seed(args.db, args.output)
    print(f"Exported {count} jobs to {args.output}")


if __name__ == "__main__":
    main()
