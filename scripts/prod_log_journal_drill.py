#!/usr/bin/env python3
"""Rotate a service log under a size cap without touching the order journal.

``events.jsonl`` stays byte for byte. The rotator does not hold the paper
order lease, and a following submit is accepted once. Stdout is one JSON
object. Research and simulated paper only.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_WORK = Path("/tmp/dipcatcher-ops17-rotate")


def _retain():
    import importlib.util

    path = _ROOT / "scripts" / "log_retain.py"
    spec = importlib.util.spec_from_file_location("dipcatcher_log_retain_ops17", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("log_retain")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dipcatcher_log_retain_ops17"] = module
    spec.loader.exec_module(module)
    return module


def execute() -> dict[str, object]:
    from quant_fund.paper.durable_book import open_book, submit_once
    from quant_fund.paper.order_lease import lock_path, release_order_lease

    if _WORK.exists():
        shutil.rmtree(_WORK)
    module = _retain()
    for stream in ("app", "proxy", "audit"):
        (_WORK / stream).mkdir(parents=True)
    service = _WORK / "app" / "service.log"
    service.write_bytes(b"x" * 50)
    journal = _WORK / "audit" / "events.jsonl"
    journal.write_bytes(b"j" * 80)
    before = journal.read_bytes()
    policy = module.Policy(10, 86400, 5)
    policies = {stream: policy for stream in ("app", "proxy", "audit")}
    report = module.retain_tree(_WORK, policies, time.time())
    rotated = _WORK / "app" / "service.log.1"
    lock = lock_path(_WORK / "book")
    untouched = not lock.exists()
    session = open_book(_WORK / "book")
    verdict = submit_once(session, _WORK / "book", "rot-1")
    again = submit_once(session, _WORK / "book", "rot-1")
    kept = journal.read_bytes() == before and journal.is_file()
    ok = bool(
        report["ok"]
        and kept
        and not service.exists()
        and rotated.read_bytes() == b"x" * 50
        and untouched
        and verdict == "accepted"
        and again == "duplicate"
        and module.sealed_name("events.jsonl")
    )
    release_order_lease(_WORK / "book")
    return {
        "duplicate": again,
        "error": "" if ok else "rotate",
        "journal_same": kept,
        "ok": ok,
        "retain_ok": report["ok"],
        "rotated": rotated.is_file(),
        "verdict": verdict,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-log-journal-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
