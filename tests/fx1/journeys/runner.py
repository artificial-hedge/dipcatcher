"""Run the frozen journey catalog and build a dipcatcher.journeys.v1 result."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tests.fx1.journeys import RESULT_SCHEMA
from tests.fx1.journeys.load import CATEGORIES, load_catalog, select_journeys, suite_sha256
from tests.fx1.journeys.support import (
    apply_setup,
    help_text,
    isolated_env,
    missing_features,
    needs_stub,
    prepare_tree,
    queue_replies,
)
from tests.fx1.journeys.textutil import FX1_KEY, LITE_KEY, redact

SCOPE = (
    "This is an automated scripted suite with a stub model and NOT the "
    "moderated five-user trial. Gate P1 stays OPEN until that trial and "
    "installed-wheel runs exist."
)
_REPO = Path(__file__).resolve().parents[3]


def run_journey(journey: dict[str, Any], root: Path) -> dict[str, Any]:
    """Run one journey. Exceptions become a failed row; nothing is rescued."""
    started = time.monotonic()
    try:
        return _run_journey(journey, root, started)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {redact(str(exc))}"
        return _row(journey, "failed", reason, started)


def run_suite(raw_filter: str = "") -> dict[str, Any]:
    """Run the selected journeys sequentially and return the result document."""
    journeys = select_journeys(load_catalog(), raw_filter)
    results: list[dict[str, Any]] = []
    for journey in journeys:
        root = Path(tempfile.mkdtemp(prefix=f"dip-{journey['id']}-"))
        try:
            result = run_journey(journey, root)
        finally:
            _remove_tree(root)
        results.append(result)
        _progress(result)
    return document(results, raw_filter=raw_filter)


def document(results: list[dict[str, Any]], *, raw_filter: str) -> dict[str, Any]:
    """Assemble the published result. Fixture keys are already redacted."""
    counts = _counts(results)
    payload = {
        "schema": RESULT_SCHEMA,
        "suite_sha256": suite_sha256(),
        "filter": raw_filter,
        "counts": counts,
        "pass_rate": _rate(counts),
        "critical": _critical(results),
        "by_category": _by_category(results),
        "journeys": results,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "tree": _tree_state(),
        "scope": SCOPE,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    return _without_keys(payload)


def gate_satisfied(payload: dict[str, Any]) -> bool:
    """True only when at least 99 journeys passed and every critical passed."""
    counts = payload["counts"]
    critical = payload["critical"]
    return bool(
        counts["n"] >= 100
        and counts["passed"] >= 99
        and critical["passed"] == critical["n"]
        and not critical["failed_ids"]
    )


def _run_journey(journey: dict[str, Any], root: Path, started: float) -> dict[str, Any]:
    shown = help_text(root)
    missing = missing_features(list(journey.get("requires", [])), shown)
    if missing:
        listed = ", ".join(missing)
        return _row(journey, "unavailable", f"required command not in /help: {listed}", started)
    paths = prepare_tree(root)
    with isolated_env(paths["config"], paths["sessions"]):
        problem = _drive(journey, paths)
    status = "failed" if problem else "passed"
    return _row(journey, status, problem or "", started)


def _drive(journey: dict[str, Any], paths: dict[str, Path]) -> str | None:
    if not needs_stub(journey):
        apply_setup(journey, paths, "")
        return _execute(journey, paths, None, "")
    from tests.fx1.stub_model_server import StubModelServer

    keys = {"fx1": FX1_KEY, "fx1-lite": LITE_KEY}
    with StubModelServer(keys) as server:
        apply_setup(journey, paths, server.url)
        setup = journey.get("setup", {})
        specs = setup.get("replies", []) if isinstance(setup, dict) else []
        queue_replies(server, list(specs), stub_url=server.url, work=str(paths["work"]))
        return _execute(journey, paths, server, server.url)


def _execute(
    journey: dict[str, Any], paths: dict[str, Path], stub: Any, stub_url: str
) -> str | None:
    if journey["mode"] == "pty":
        from tests.fx1.journeys.pty_run import run_pty

        problem, _transcript, _code, _ended = run_pty(journey, paths, stub=stub, stub_url=stub_url)
        return problem
    from tests.fx1.journeys.inproc import run_inproc

    problem, _transcript, _code, _ended = run_inproc(journey, paths, stub=stub, stub_url=stub_url)
    return problem


def _row(journey: dict[str, Any], status: str, reason: str, started: float) -> dict[str, Any]:
    cleaned = redact(reason)
    return {
        "id": journey["id"],
        "status": status,
        "reason": cleaned,
        "attribution": _attribution(status, cleaned),
        "duration_s": round(time.monotonic() - started, 3),
        "category": journey["category"],
        "critical": journey["critical"],
        "mode": journey["mode"],
    }


def _attribution(status: str, reason: str) -> str:
    if status == "passed":
        return ""
    if status == "unavailable":
        return "not-yet-implemented feature"
    if "not quarantined" in reason:
        return "product defect"
    return "unclassified"


def _counts(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "n": len(results),
        "passed": _status_count(results, "passed"),
        "failed": _status_count(results, "failed"),
        "unavailable": _status_count(results, "unavailable"),
    }


def _status_count(results: list[dict[str, Any]], status: str) -> int:
    return sum(1 for row in results if row["status"] == status)


def _rate(counts: dict[str, int]) -> float:
    if counts["n"] == 0:
        return 0.0
    return counts["passed"] / counts["n"]


def _critical(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in results if row["critical"]]
    failed = [row["id"] for row in rows if row["status"] != "passed"]
    return {"n": len(rows), "passed": len(rows) - len(failed), "failed_ids": failed}


def _by_category(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for category in CATEGORIES:
        rows = [row for row in results if row["category"] == category]
        summary[category] = {
            "n": len(rows),
            "passed": _status_count(rows, "passed"),
            "failed": _status_count(rows, "failed"),
            "unavailable": _status_count(rows, "unavailable"),
        }
    return summary


def _tree_state() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    lines = [line for line in status.splitlines() if line.strip()]
    return {"head": head, "dirty": bool(lines), "status": lines[:40]}


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _without_keys(payload: dict[str, Any]) -> dict[str, Any]:
    import json

    text = json.dumps(payload)
    if FX1_KEY not in text and LITE_KEY not in text:
        return payload
    cleaned = json.loads(redact(text))
    if isinstance(cleaned, dict):
        return cleaned
    return payload


def _progress(result: dict[str, Any]) -> None:
    reason = result["reason"]
    suffix = f" {reason}" if reason else ""
    print(
        f"{result['id']} {result['status']} {result['duration_s']}s{suffix}",
        file=sys.stderr,
        flush=True,
    )


def _remove_tree(root: Path) -> None:
    import shutil

    shutil.rmtree(root, ignore_errors=True)
