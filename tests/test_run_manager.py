import time

import pytest

from src.run_manager import RunManager, RunStore


def _wait_for_status(manager, run_id, owner, terminal, timeout=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = manager.get_for_owner(run_id, owner)
        if record["status"] in terminal:
            return record
        time.sleep(0.02)
    raise AssertionError("run did not reach terminal status")


def test_run_manager_persists_result_and_reuses_owner_cache(tmp_path):
    manager = RunManager(store=RunStore(tmp_path / "runs.db"), max_workers=1)
    first = manager.submit(
        "alice",
        "request-hash",
        lambda run_id, deadline: {"workflow_status": "completed", "run_id": run_id},
    )
    completed = _wait_for_status(manager, first["run_id"], "alice", {"completed"})
    assert completed["result"]["workflow_status"] == "completed"

    cached = manager.submit(
        "alice",
        "request-hash",
        lambda run_id, deadline: {"workflow_status": "failed"},
    )
    assert cached["status"] == "completed"
    assert cached["cache_hit"] is True


def test_run_manager_enforces_owner_isolation(tmp_path):
    manager = RunManager(store=RunStore(tmp_path / "runs.db"), max_workers=1)
    record = manager.submit(
        "alice",
        "hash",
        lambda run_id, deadline: {"workflow_status": "completed"},
    )
    with pytest.raises(PermissionError):
        manager.get_for_owner(record["run_id"], "bob")


def test_run_manager_marks_late_result_as_timed_out(tmp_path):
    manager = RunManager(store=RunStore(tmp_path / "runs.db"), max_workers=1)

    def slow_task(run_id, deadline):
        time.sleep(0.05)
        return {"workflow_status": "completed"}

    record = manager.submit(
        "alice",
        "slow",
        slow_task,
        timeout_seconds=0.01,
        allow_cache=False,
    )
    completed = _wait_for_status(
        manager,
        record["run_id"],
        "alice",
        {"timed_out"},
    )
    assert completed["result"]["workflow_status"] == "timed_out"
