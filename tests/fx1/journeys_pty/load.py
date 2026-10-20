"""Load and check the frozen journey catalog."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tests.fx1.journeys_pty import SCHEMA

CATALOG_PATH = Path(__file__).resolve().parent / "journeys.json"
CATEGORIES = (
    "onboarding",
    "model-selection",
    "data-discovery",
    "research-planning",
    "execution",
    "follow-ups",
    "verification",
    "cancellation",
    "recovery",
    "export",
)
FEATURES = ("data", "runs", "models-check")
MODES = ("inproc", "pty")
_ID_RE = re.compile(r"^J\d{3}$")
_STEP_KEYS = {
    "say",
    "secret",
    "url",
    "expect",
    "forbid",
    "wait",
    "signal",
    "resume",
    "hang_s",
}
_FINAL_KEYS = {
    "exit_code",
    "ended",
    "session_forbids",
    "session_contains",
    "export_forbids",
    "export_contains",
    "history_forbids",
    "telemetry_forbids",
    "transcript_forbids",
    "job_state",
    "corrupt_quarantined",
    "session_exists",
}
_WAITS = {"idle", "job_done", "request", "exit"}
_SIGNALS = {"INT", "INT2", "KILL"}


def suite_sha256(path: Path | None = None) -> str:
    """SHA-256 of the frozen catalog bytes."""
    target = CATALOG_PATH if path is None else path
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_catalog(path: Path | None = None) -> list[dict[str, Any]]:
    """Return validated journeys in id order."""
    target = CATALOG_PATH if path is None else path
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("catalog schema must be dipcatcher.journey.v1")
    rows = payload.get("journeys")
    if not isinstance(rows, list):
        raise ValueError("catalog journeys must be a list")
    checked = [_journey(row) for row in rows]
    _suite(checked)
    return checked


def select_journeys(journeys: list[dict[str, Any]], raw_filter: str) -> list[dict[str, Any]]:
    """Keep journeys whose id, category, mode, or ``critical`` matches."""
    tokens = [part.strip() for part in raw_filter.split(",") if part.strip()]
    if not tokens:
        return list(journeys)
    chosen = [row for row in journeys if _matches(row, tokens)]
    if not chosen:
        raise ValueError(f"JOURNEYS_FILTER matched nothing: {raw_filter}")
    return chosen


def _matches(row: dict[str, Any], tokens: list[str]) -> bool:
    haystack = {row["id"], row["category"], row["mode"]}
    if row["critical"]:
        haystack.add("critical")
    return any(token in haystack for token in tokens)


def _suite(rows: list[dict[str, Any]]) -> None:
    ids = [row["id"] for row in rows]
    if ids != [f"J{number:03d}" for number in range(1, 101)]:
        raise ValueError("catalog must be exactly J001..J100 in order")
    for category in CATEGORIES:
        count = sum(1 for row in rows if row["category"] == category)
        if count != 10:
            raise ValueError(f"{category} has {count} journeys, expected 10")
    critical = [row for row in rows if row["critical"]]
    if len(critical) < 20:
        raise ValueError("need at least 20 critical journeys")
    pty = [row for row in rows if row["mode"] == "pty"]
    if len(pty) < 12:
        raise ValueError("need at least 12 pty journeys")
    if any(row["mode"] != "pty" for row in critical):
        raise ValueError("every critical journey must be pty")
    lab = sum(1 for row in rows if row["executes_lab"])
    if lab > 15:
        raise ValueError(f"{lab} journeys execute a lab capability; max is 15")


def _journey(row: object) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("journey must be an object")
    ident = _identity(row)
    _flags(ident, row)
    _steps_ok(ident, row)
    _final_ok(ident, row)
    return row


def _identity(row: dict[str, Any]) -> str:
    ident = str(row.get("id", ""))
    if _ID_RE.fullmatch(ident) is None:
        raise ValueError(f"bad journey id {ident!r}")
    _require_choice(ident, row, "category", CATEGORIES)
    _require_choice(ident, row, "mode", MODES)
    if row.get("schema") != SCHEMA:
        raise ValueError(f"{ident} schema must be {SCHEMA}")
    title = row.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{ident} needs a title")
    return ident


def _flags(ident: str, row: dict[str, Any]) -> None:
    if not isinstance(row.get("critical"), bool):
        raise ValueError(f"{ident} critical must be a bool")
    if not isinstance(row.get("executes_lab"), bool):
        raise ValueError(f"{ident} executes_lab must be a bool")
    requires = _str_list(row.get("requires", []), ident, "requires")
    unknown = [item for item in requires if item not in FEATURES]
    if unknown:
        raise ValueError(f"{ident} unknown requires {unknown}")
    if row["category"] == "onboarding" and _endpoints(row):
        raise ValueError(f"{ident} onboarding cannot preset endpoints")


def _steps_ok(ident: str, row: dict[str, Any]) -> None:
    steps = row.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{ident} needs steps")
    checked = [_step(ident, step) for step in steps]
    if any(_control(step) for step in checked) and row["mode"] != "pty":
        raise ValueError(f"{ident} signal/resume steps require pty")


def _final_ok(ident: str, row: dict[str, Any]) -> None:
    final = row.get("final", {})
    if not isinstance(final, dict):
        raise ValueError(f"{ident} final must be an object")
    extra = set(final) - _FINAL_KEYS
    if extra:
        raise ValueError(f"{ident} unknown final keys {sorted(extra)}")


def _endpoints(row: dict[str, Any]) -> list[str]:
    setup = row.get("setup", {})
    if not isinstance(setup, dict):
        raise ValueError(f"{row.get('id')} setup must be an object")
    raw = setup.get("endpoints", [])
    if not isinstance(raw, list):
        raise ValueError(f"{row.get('id')} endpoints must be a list")
    return [str(item) for item in raw]


def _require_choice(ident: str, row: dict[str, Any], key: str, allowed: tuple[str, ...]) -> None:
    if row.get(key) not in allowed:
        raise ValueError(f"{ident} {key} must be one of {allowed}")


def _step(ident: str, step: object) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError(f"{ident} step must be an object")
    extra = set(step) - _STEP_KEYS
    if extra:
        raise ValueError(f"{ident} unknown step keys {sorted(extra)}")
    wait = step.get("wait", "idle")
    if wait not in _WAITS:
        raise ValueError(f"{ident} bad wait {wait!r}")
    signal = step.get("signal")
    if signal is not None and signal not in _SIGNALS:
        raise ValueError(f"{ident} bad signal {signal!r}")
    for key in ("expect", "forbid"):
        _str_list(step.get(key, []), ident, key)
    for key in ("say", "secret", "url"):
        value = step.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{ident} {key} must be a string")
    if not any(step.get(key) for key in ("say", "signal", "resume")):
        raise ValueError(f"{ident} step needs say, signal, or resume")
    return step


def _control(step: dict[str, Any]) -> bool:
    return bool(step.get("signal") or step.get("resume"))


def _str_list(value: object, ident: str, key: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{ident} {key} must be a list of strings")
    return list(value)
