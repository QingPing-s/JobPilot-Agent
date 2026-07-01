from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .job_recorder import extract_job_record, format_job_record_as_jd
from .schemas import JobPosting

DEFAULT_DATA_DIR = Path(os.getenv("JOBPILOT_DATA_DIR", "data"))
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "jobpilot.db"


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
            content_hash TEXT,
            parsed_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "content_hash" not in existing_columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN content_hash TEXT")
    if "parsed_json" not in existing_columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN parsed_json TEXT")
    conn.commit()


def _content_hash(raw_text: str) -> str:
    import hashlib

    return hashlib.sha1(raw_text.strip().encode("utf-8")).hexdigest()


def _model_to_dict(model: JobPosting) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _validate_job_posting(data: dict[str, Any]) -> dict[str, Any]:
    if hasattr(JobPosting, "model_validate"):
        return _model_to_dict(JobPosting.model_validate(data))
    return _model_to_dict(JobPosting.parse_obj(data))


def _record_to_parsed_job(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a saved job record to the JobPosting shape used by the graph."""
    parsed = {
        "job_id": record["job_id"],
        "title": record.get("title") or "未命名岗位",
        "company": record.get("company") or "",
        "location": record.get("location"),
        "employment_type": "实习" if "实习" in f"{record.get('title', '')}\n{record.get('raw_text', '')}" else None,
        "salary": record.get("salary"),
        "responsibilities": record.get("responsibilities") if isinstance(record.get("responsibilities"), list) else [],
        "required_skills": record.get("required_skills") if isinstance(record.get("required_skills"), list) else [],
        "preferred_skills": record.get("preferred_skills") if isinstance(record.get("preferred_skills"), list) else [],
        "education_requirement": record.get("education"),
        "experience_requirement": record.get("duration"),
        "source_url": None,
        "raw_text": record.get("raw_text") or "",
    }
    return _validate_job_posting(parsed)


def _row_to_fallback_parsed_job(row: sqlite3.Row) -> dict[str, Any]:
    """Return a minimal JobPosting for legacy rows that do not have parsed_json yet."""
    return _validate_job_posting(
        {
            "job_id": row["job_id"],
            "title": row["title"],
            "company": row["company"],
            "location": row["location"],
            "employment_type": "实习" if "实习" in f"{row['title']}\n{row['raw_text']}" else None,
            "salary": row["salary"],
            "responsibilities": [],
            "required_skills": [],
            "preferred_skills": [],
            "education_requirement": row["education"],
            "experience_requirement": row["duration"],
            "source_url": None,
            "raw_text": row["raw_text"],
        }
    )


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    parsed_json = row["parsed_json"] if "parsed_json" in row.keys() else None
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
        "content_hash": row["content_hash"] if "content_hash" in row.keys() else None,
        "has_parsed_job": bool(parsed_json),
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
    parsed_job = _record_to_parsed_job(record)
    parsed_json = json.dumps(parsed_job, ensure_ascii=False)
    content_hash = _content_hash(record["raw_text"])
    now = _utc_now()

    with _connect(db_path) as conn:
        existing = conn.execute("SELECT created_at FROM jobs WHERE job_id = ?", (record["job_id"],)).fetchone()
        created_at = existing["created_at"] if existing else now
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, title, company, location, salary, duration, education, raw_text,
                source, source_filename, content_hash, parsed_json, created_at, updated_at, is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
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
                content_hash = excluded.content_hash,
                parsed_json = excluded.parsed_json,
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
                content_hash,
                parsed_json,
                created_at,
                now,
            ),
        )
        conn.commit()

    return {
        **record,
        "parsed_job": parsed_job,
        "content_hash": content_hash,
        "has_parsed_job": True,
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


def list_parsed_jobs(
    db_path: str | Path = DEFAULT_DB_PATH,
    include_inactive: bool = False,
) -> list[dict[str, Any]]:
    """Load active jobs in JobPosting format without reparsing JD text."""
    query = "SELECT * FROM jobs"
    params: tuple[Any, ...] = ()
    if not include_inactive:
        query += " WHERE is_active = ?"
        params = (1,)
    query += " ORDER BY updated_at DESC"

    parsed_jobs: list[dict[str, Any]] = []
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    for row in rows:
        parsed_json = row["parsed_json"] if "parsed_json" in row.keys() else None
        if parsed_json:
            try:
                data = json.loads(parsed_json)
                if isinstance(data, dict):
                    data["job_id"] = data.get("job_id") or row["job_id"]
                    data["raw_text"] = data.get("raw_text") or row["raw_text"]
                    parsed_jobs.append(_validate_job_posting(data))
                    continue
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        parsed_jobs.append(_row_to_fallback_parsed_job(row))
    return parsed_jobs


def backfill_parsed_jobs(db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Populate parsed_json for legacy rows once, using the local lightweight parser."""
    updated = 0
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE parsed_json IS NULL OR parsed_json = ''"
        ).fetchall()
        for row in rows:
            record = extract_job_record(
                raw_text=row["raw_text"],
                filename=row["source_filename"],
                source=row["source"] or "legacy",
            )
            record.update(
                {
                    "job_id": row["job_id"],
                    "title": row["title"],
                    "company": row["company"],
                    "location": row["location"],
                    "salary": row["salary"],
                    "duration": row["duration"],
                    "education": row["education"],
                }
            )
            parsed_job = _record_to_parsed_job(record)
            conn.execute(
                "UPDATE jobs SET content_hash = ?, parsed_json = ?, updated_at = ? WHERE job_id = ?",
                (
                    _content_hash(row["raw_text"]),
                    json.dumps(parsed_job, ensure_ascii=False),
                    _utc_now(),
                    row["job_id"],
                ),
            )
            updated += 1
        conn.commit()
    return updated


def soft_delete_job(job_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> bool:
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE jobs SET is_active = 0, updated_at = ? WHERE job_id = ? AND is_active = 1",
            (_utc_now(), job_id),
        )
        conn.commit()
    return cursor.rowcount > 0


def restore_job(job_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> bool:
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE jobs SET is_active = 1, updated_at = ? WHERE job_id = ? AND is_active = 0",
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
