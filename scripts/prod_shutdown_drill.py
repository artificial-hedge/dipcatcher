#!/usr/bin/env python3
"""SIGTERM, SIGINT, and SIGKILL a simulated paper order loop.

SIGTERM and SIGINT finish the in-flight order, exit 0, and resume with the
same ledger digest. SIGKILL lands after the temporary ledger bytes exist and
before they are published. Restart applies the journaled intent once. A second
restart keeps that digest. Stdout is one JSON object. Research and simulated
paper only.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from quant_fund.paper.durable_book import (
    file_digest,
    ledger_intact,
    ledger_order_count,
    ledger_path,
    pending_order_ids,
)

_ROOT = Path(__file__).resolve().parents[1]
_FLOW = _ROOT / "scripts" / "paper_flow.py"
_WORK = Path("/tmp/dipcatcher-ops16-shutdown")


def _spawn(state: Path, count: int, prefix: str, env: dict[str, str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(_FLOW), str(state), str(count), prefix],
        cwd=_ROOT,
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _reap(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    try:
        _out, err = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        return ""
    return err or ""


def _wait_file(proc: subprocess.Popen[str], path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def _wait_commit(proc: subprocess.Popen[str], state: Path, timeout: float) -> bool:
    journal = state / "journal" / "events.jsonl"
    needle = b'"kind": "commit"'
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if journal.is_file() and needle in journal.read_bytes():
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("DIPCATCHER_REPLACE_HOLD", None)
    return env


def _fail(error: str, exit_code: int | None, detail: str) -> dict[str, object]:
    return {
        "detail": detail[-400:],
        "digest_match": False,
        "error": error,
        "exit": exit_code,
        "ok": False,
    }


def _graceful(state: Path, sig: int, prefix: str) -> dict[str, object]:
    env = _clean_env()
    proc = _spawn(state, 100_000, prefix, env)
    resume: subprocess.Popen[str] | None = None
    try:
        if not _wait_file(proc, state / "ready", 90) or not _wait_commit(proc, state, 90):
            return _fail("not_flowing", proc.poll(), _reap(proc))
        os.kill(proc.pid, sig)
        code = proc.wait(timeout=90)
        detail = _reap(proc)
        path = ledger_path(state)
        if code != 0 or not ledger_intact(path):
            return _fail("exit_or_torn", code, detail)
        pending = pending_order_ids(state)
        before = file_digest(path)
        resume = _spawn(state, 0, prefix, env)
        resume_code = resume.wait(timeout=90)
        resume_detail = _reap(resume)
        after = file_digest(path)
        match = resume_code == 0 and before == after and pending == [] and ledger_intact(path)
        return {
            "detail": (detail + resume_detail)[-400:],
            "digest_match": match,
            "error": "" if match else "resume",
            "exit": code,
            "ledger_sha256": before,
            "ok": match,
            "pending": len(pending),
            "resume_exit": resume_code,
        }
    finally:
        _reap(proc)
        if resume is not None:
            _reap(resume)


def _resume_pair(state: Path, env: dict[str, str]) -> tuple[int, int, str]:
    first = _spawn(state, 0, "kill", env)
    first_code = first.wait(timeout=90)
    first_detail = _reap(first)
    second = _spawn(state, 0, "kill", env)
    second_code = second.wait(timeout=90)
    second_detail = _reap(second)
    return first_code, second_code, first_detail + second_detail


def _kill(state: Path, hold: Path) -> dict[str, object]:
    env = _clean_env()
    env["DIPCATCHER_REPLACE_HOLD"] = str(hold)
    env["DIPCATCHER_REPLACE_HOLD_SKIP"] = "1"
    proc = _spawn(state, 5, "kill", env)
    try:
        if not _wait_file(proc, hold, 90):
            return _fail("no_hold", proc.poll(), _reap(proc))
        path = ledger_path(state)
        before = file_digest(path)
        os.kill(proc.pid, signal.SIGKILL)
        code = proc.wait(timeout=30)
        detail = _reap(proc)
        intact = ledger_intact(path) and file_digest(path) == before
        pending = pending_order_ids(state)
        clean = _clean_env()
        first_code, second_code, resume_detail = _resume_pair(state, clean)
        after = file_digest(path)
        third = _spawn(state, 0, "kill", clean)
        third_code = third.wait(timeout=90)
        third_detail = _reap(third)
        stable = file_digest(path) == after and third_code == 0
        once = len(pending) == 1 and ledger_order_count(path, pending[0]) == 1
        ok = (
            code == -signal.SIGKILL
            and intact
            and first_code == 0
            and second_code == 0
            and once
            and stable
            and pending_order_ids(state) == []
        )
        return {
            "detail": (detail + resume_detail + third_detail)[-400:],
            "error": "" if ok else "kill_resume",
            "exit": code,
            "ledger_intact": intact,
            "ledger_sha256": before,
            "ok": ok,
            "once": once,
            "pending_before": len(pending),
            "resume_exit": first_code,
            "resume_sha256": after,
            "stable": stable,
        }
    finally:
        _reap(proc)


def execute() -> dict[str, object]:
    if _WORK.exists():
        shutil.rmtree(_WORK)
    _WORK.mkdir(parents=True)
    term = _graceful(_WORK / "term", signal.SIGTERM, "term")
    intr = _graceful(_WORK / "intr", signal.SIGINT, "intr")
    killed = _kill(_WORK / "kill", _WORK / "hold")
    ok = bool(term["ok"] and intr["ok"] and killed["ok"])
    error = ""
    for label, report in (("term", term), ("intr", intr), ("kill", killed)):
        if not report["ok"] and isinstance(report.get("error"), str):
            error = f"{label}:{report['error']}"
            break
    return {"error": error, "intr": intr, "kill": killed, "ok": ok, "term": term}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-shutdown-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
