from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .run_control import cancel_run, register_run, unregister_run

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "timed_out", "awaiting_review"}
DEFAULT_RUN_DB_PATH = Path(
    os.getenv("JOBPILOT_RUN_DB", Path(os.getenv("JOBPILOT_DATA_DIR", "data")) / "jobpilot_runs.db")
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, db_path: str | Path = DEFAULT_RUN_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_owner_hash ON runs(owner_id, request_hash, status)"
            )
            connection.commit()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
        result = None
        if row["result_json"]:
            try:
                result = json.loads(row["result_json"])
            except json.JSONDecodeError:
                result = None
        return {
            "run_id": row["run_id"],
            "owner_id": row["owner_id"],
            "status": row["status"],
            "request_hash": row["request_hash"],
            "cache_hit": bool(row["cache_hit"]),
            "result": result,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create(
        self,
        run_id: str,
        owner_id: str,
        request_hash: str,
        *,
        status: str = "queued",
        cache_hit: bool = False,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        result_json = json.dumps(result, ensure_ascii=False, default=str) if result is not None else None
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, owner_id, status, request_hash, cache_hit,
                    result_json, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (run_id, owner_id, status, request_hash, int(cache_hit), result_json, now, now),
            )
            connection.commit()
        return self.get(run_id) or {}

    def update(
        self,
        run_id: str,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        assignments = ["updated_at = ?"]
        values: list[Any] = [_utc_now()]
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if result is not None:
            assignments.append("result_json = ?")
            values.append(json.dumps(result, ensure_ascii=False, default=str))
        if error is not None:
            assignments.append("error = ?")
            values.append(error)
        values.append(run_id)
        with self._lock, self._connect() as connection:
            connection.execute(f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?", values)
            connection.commit()
        return self.get(run_id)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def find_cached(self, owner_id: str, request_hash: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM runs
                WHERE owner_id = ? AND request_hash = ? AND status = 'completed'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (owner_id, request_hash),
            ).fetchone()
        return self._row_to_record(row) if row else None


RunCallable = Callable[[str, float | None], dict[str, Any]]


class RunManager:
    def __init__(
        self,
        store: RunStore | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.store = store or RunStore()
        workers = max_workers or int(os.getenv("JOBPILOT_WORKER_THREADS", "2"))
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="jobpilot")
        self._futures: dict[str, Future] = {}
        self._lock = threading.RLock()

    def submit(
        self,
        owner_id: str,
        request_hash: str,
        task: RunCallable,
        *,
        timeout_seconds: float = 300.0,
        allow_cache: bool = True,
    ) -> dict[str, Any]:
        if allow_cache:
            cached = self.store.find_cached(owner_id, request_hash)
            if cached and isinstance(cached.get("result"), dict):
                run_id = uuid4().hex
                return self.store.create(
                    run_id,
                    owner_id,
                    request_hash,
                    status="completed",
                    cache_hit=True,
                    result=cached["result"],
                )

        run_id = uuid4().hex
        record = self.store.create(run_id, owner_id, request_hash)
        future = self.executor.submit(
            self._execute,
            run_id,
            task,
            timeout_seconds,
        )
        with self._lock:
            self._futures[run_id] = future
        return record

    def _execute(
        self,
        run_id: str,
        task: RunCallable,
        timeout_seconds: float,
    ) -> None:
        control = register_run(run_id, timeout_seconds)
        self.store.update(run_id, status="running")
        try:
            result = task(run_id, control.deadline_epoch)
            status = str(result.get("workflow_status") or "completed")
            if control.cancel_event.is_set():
                status = "cancelled"
            elif control.deadline_epoch is not None and time.time() >= control.deadline_epoch:
                status = "timed_out"
                result["workflow_status"] = "timed_out"
            elif status not in TERMINAL_STATUSES:
                status = "completed"
            self.store.update(run_id, status=status, result=result)
        except Exception as exc:
            status = "cancelled" if control.cancel_event.is_set() else "failed"
            self.store.update(run_id, status=status, error=str(exc))
        finally:
            unregister_run(run_id)
            with self._lock:
                self._futures.pop(run_id, None)

    def resume(
        self,
        run_id: str,
        owner_id: str,
        task: RunCallable,
        *,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        record = self.get_for_owner(run_id, owner_id, is_admin=True)
        if record["status"] != "awaiting_review":
            raise ValueError("只有等待人工确认的任务可以继续。")
        future = self.executor.submit(self._execute, run_id, task, timeout_seconds)
        with self._lock:
            self._futures[run_id] = future
        return self.store.update(run_id, status="queued") or record

    def cancel(self, run_id: str, owner_id: str, *, is_admin: bool = False) -> dict[str, Any]:
        record = self.get_for_owner(run_id, owner_id, is_admin=is_admin)
        if record["status"] in TERMINAL_STATUSES:
            return record
        cancel_run(run_id)
        with self._lock:
            future = self._futures.get(run_id)
            cancelled_before_start = bool(future and future.cancel())
        status = "cancelled" if cancelled_before_start else "cancelling"
        return self.store.update(run_id, status=status) or record

    def get_for_owner(self, run_id: str, owner_id: str, *, is_admin: bool = False) -> dict[str, Any]:
        record = self.store.get(run_id)
        if record is None:
            raise KeyError(run_id)
        if not is_admin and record["owner_id"] != owner_id:
            raise PermissionError(run_id)
        return record
