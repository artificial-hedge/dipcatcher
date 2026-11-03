#!/usr/bin/env python3
"""Snapshot a running paper book, restore it, and reject a stale copy.

The worker keeps submitting simulated orders. The snapshot reads each state
file once while that process is still in the loop, then ``backup_paper_state``
archives that frozen copy. Restore onto a fresh directory reproduces the
ledger, journal, and audit digests. An order the snapshot already committed
is a duplicate. Restoring the archive onto the newer live directory fails
closed. Stdout is one JSON object.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from quant_fund.paper.durable_book import (
    content_digests,
    ledger_intact,
    ledger_order_count,
    ledger_path,
    open_book,
    submit_once,
)
from quant_fund.paper.state_backup import PaperBackupError, backup_paper_state, restore_paper_state

_ROOT = Path(__file__).resolve().parents[1]
_FLOW = _ROOT / "scripts" / "paper_flow.py"
_WORK = Path("/tmp/dipcatcher-ops16-backup")
_RELATIVE = ("broker_state.json", "journal/events.jsonl", "audit/audit.jsonl")


def _spawn(state: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.pop("DIPCATCHER_REPLACE_HOLD", None)
    return subprocess.Popen(
        [sys.executable, str(_FLOW), str(state), "100000", "snap"],
        cwd=_ROOT,
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _reap(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        return


def _commits(raw: bytes) -> int:
    count = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(payload, dict) and payload.get("kind") == "commit":
            count += 1
    return count


def _wait_commits(proc: subprocess.Popen[str], state: Path, count: int) -> bool:
    deadline = time.monotonic() + 90
    journal = state / "journal" / "events.jsonl"
    while time.monotonic() < deadline:
        if journal.is_file() and _commits(journal.read_bytes()) >= count:
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def _read_tree(state: Path) -> dict[str, bytes] | None:
    found: dict[str, bytes] = {}
    for relative in _RELATIVE:
        path = state / relative
        if not path.is_file():
            return None
        found[relative] = path.read_bytes()
    if not _ledger_bytes_ok(found["broker_state.json"]):
        return None
    return found


def _ledger_bytes_ok(raw: bytes) -> bool:
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("schema") == "paper_broker_state.v1"


def _capture(proc: subprocess.Popen[str], state: Path, frozen: Path) -> dict[str, bytes] | None:
    deadline = time.monotonic() + 30
    found: dict[str, bytes] | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None
        found = _read_tree(state)
        if found is not None and _commits(found["journal/events.jsonl"]) >= 2:
            break
        time.sleep(0.05)
    else:
        return None
    if found is None:
        return None
    if frozen.exists():
        shutil.rmtree(frozen)
    for relative, raw in found.items():
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return found


def _digests(raw: dict[str, bytes]) -> dict[str, str]:
    return {relative: hashlib.sha256(blob).hexdigest() for relative, blob in raw.items()}


def _stop(proc: subprocess.Popen[str]) -> int:
    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGTERM)
    code = proc.wait(timeout=90)
    _reap(proc)
    return code


def _replay(fresh: Path) -> tuple[str, str, str, bool]:
    session = open_book(fresh)
    replay = submit_once(session, fresh, "snap-0")
    again = submit_once(session, fresh, "snap-0")
    created = submit_once(session, fresh, "snap-new")
    path = ledger_path(fresh)
    once = ledger_order_count(path, "snap-0") == 1 and ledger_order_count(path, "snap-new") == 1
    return replay, again, created, once


def _stale(archive: Path, live: Path, live_digests: dict[str, str]) -> tuple[str, bool]:
    try:
        restore_paper_state(archive, live)
    except PaperBackupError as exc:
        return str(exc), content_digests(live) == live_digests
    return "accepted", False


def _advance(proc: subprocess.Popen[str], state: Path, seen: int) -> bool:
    return _wait_commits(proc, state, seen + 1)


def execute() -> dict[str, object]:
    if _WORK.exists():
        shutil.rmtree(_WORK)
    live = _WORK / "live"
    live.mkdir(parents=True)
    proc = _spawn(live)
    frozen = _WORK / "frozen"
    try:
        captured = _capture(proc, live, frozen)
        if captured is None:
            return {"error": "snapshot", "ok": False}
        seen = _commits(captured["journal/events.jsonl"])
        if not _advance(proc, live, seen):
            return {"error": "did_not_advance", "ok": False}
        exit_code = _stop(proc)
    finally:
        _reap(proc)
    if exit_code != 0:
        return {"error": f"exit_{exit_code}", "ok": False}
    archive = _WORK / "archive"
    backup_paper_state(frozen, archive)
    snapshot = _digests(captured)
    live_digests = content_digests(live)
    fresh = _WORK / "fresh"
    restore_paper_state(archive, fresh)
    restored = content_digests(fresh)
    replay, again, created, once = _replay(fresh)
    rejected, unchanged = _stale(archive, live, live_digests)
    ok = _ok(snapshot, restored, live_digests, replay, again, created, once, rejected, unchanged)
    return {
        "digests_match": restored == snapshot,
        "duplicate": again,
        "error": "" if ok else "restore",
        "live_advanced": live_digests != snapshot,
        "live_digests": live_digests,
        "new_order": created,
        "ok": ok,
        "replay": replay,
        "restored_digests": restored,
        "snapshot_digests": snapshot,
        "stale_rejected": rejected,
        "unchanged": unchanged,
    }


def _ok(
    snapshot: dict[str, str],
    restored: dict[str, str],
    live_digests: dict[str, str],
    replay: str,
    again: str,
    created: str,
    once: bool,
    rejected: str,
    unchanged: bool,
) -> bool:
    return (
        restored == snapshot
        and live_digests != snapshot
        and replay == "duplicate"
        and again == "duplicate"
        and created == "accepted"
        and once
        and rejected == "dest_differs"
        and unchanged
        and ledger_intact(ledger_path(_WORK / "fresh"))
    )


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-backup-restore-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
