from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunControl:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    deadline_epoch: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)


_CONTROLS: dict[str, RunControl] = {}
_LOCK = threading.RLock()


def register_run(run_id: str, timeout_seconds: float | None = None) -> RunControl:
    deadline = time.time() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    control = RunControl(deadline_epoch=deadline)
    with _LOCK:
        _CONTROLS[run_id] = control
    return control


def get_control(run_id: str) -> RunControl | None:
    with _LOCK:
        return _CONTROLS.get(run_id)


def cancel_run(run_id: str) -> bool:
    control = get_control(run_id)
    if control is None:
        return False
    control.cancel_event.set()
    publish_event(run_id, {"event_type": "control", "status": "cancelling"})
    return True


def is_cancelled(run_id: str) -> bool:
    control = get_control(run_id)
    return bool(control and control.cancel_event.is_set())


def deadline_exceeded(run_id: str) -> bool:
    control = get_control(run_id)
    return bool(control and control.deadline_epoch and time.time() >= control.deadline_epoch)


def publish_event(run_id: str, event: dict[str, Any]) -> None:
    control = get_control(run_id)
    if control is None:
        return
    with control.condition:
        control.events.append(dict(event))
        control.condition.notify_all()


def events_since(run_id: str, offset: int) -> tuple[list[dict[str, Any]], int]:
    control = get_control(run_id)
    if control is None:
        return [], offset
    with control.condition:
        events = [dict(event) for event in control.events[offset:]]
        return events, len(control.events)


def unregister_run(run_id: str) -> None:
    with _LOCK:
        _CONTROLS.pop(run_id, None)
