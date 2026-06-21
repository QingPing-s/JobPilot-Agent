from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .job_recorder import extract_job_record, format_job_record_as_jd


DEFAULT_DB_PATH = Path("data/jobpilot.db")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_job_store(conn)
    return conn


def init_job_store(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            location TEXT,
            salary TEXT,
            duration TEXT,
            education TEXT,
            raw_text TEXT NOT NULL,
            source TEXT,
            source_filename TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.commit()


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "title": row["title"],
        "company": row["company"],
        "location": row["location"],
        "salary": row["salary"],
        "duration": row["duration"],
        "education": row["education"],
        "raw_text": row["raw_text"],
        "source": row["source"],
        "source_filename": row["source_filename"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "is_active": bool(row["is_active"]),
    }


def upsert_job(
    raw_text: str,
    filename: str | None = None,
    source: str = "frontend",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    record = extract_job_record(raw_text=raw_text, filename=filename, source=source)
    now = _utc_now()

    with _connect(db_path) as conn:
        existing = conn.execute("SELECT created_at FROM jobs WHERE job_id = ?", (record["job_id"],)).fetchone()
        created_at = existing["created_at"] if existing else now
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, title, company, location, salary, duration, education, raw_text,
                source, source_filename, created_at, updated_at, is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(job_id) DO UPDATE SET
                title = excluded.title,
                company = excluded.company,
                location = excluded.location,
                salary = excluded.salary,
                duration = excluded.duration,
                education = excluded.education,
                raw_text = excluded.raw_text,
                source = excluded.source,
                source_filename = excluded.source_filename,
                updated_at = excluded.updated_at,
                is_active = 1
            """,
            (
                record["job_id"],
                record["title"],
                record["company"],
                record.get("location"),
                record.get("salary"),
                record.get("duration"),
                record.get("education"),
                record["raw_text"],
                record.get("source"),
                record.get("source_filename"),
                created_at,
                now,
            ),
        )
        conn.commit()

    return {
        **record,
        "created_at": created_at,
        "updated_at": now,
        "is_active": True,
        "already_exists": bool(existing),
    }


def upsert_jobs(
    jd_texts: list[str],
    jd_filenames: list[str] | None = None,
    source: str = "frontend",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    filenames = jd_filenames or []
    saved = []
    for index, raw_text in enumerate(jd_texts):
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        if not text:
            continue
        filename = filenames[index] if index < len(filenames) else None
        saved.append(upsert_job(raw_text=text, filename=filename, source=source, db_path=db_path))
    return saved


def list_jobs(
    db_path: str | Path = DEFAULT_DB_PATH,
    include_inactive: bool = False,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM jobs"
    params: tuple[Any, ...] = ()
    if not include_inactive:
        query += " WHERE is_active = ?"
        params = (1,)
    query += " ORDER BY updated_at DESC"

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_job(row) for row in rows]


def soft_delete_job(job_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> bool:
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE jobs SET is_active = 0, updated_at = ? WHERE job_id = ? AND is_active = 1",
            (_utc_now(), job_id),
        )
        conn.commit()
    return cursor.rowcount > 0


def export_active_jobs_to_jd_folder(
    jd_dir: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[Path]:
    folder = Path(jd_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for job in list_jobs(db_path=db_path, include_inactive=False):
        path = folder / f"{job['job_id']}.txt"
        path.write_text(format_job_record_as_jd(job), encoding="utf-8")
        paths.append(path)
    return paths
