#!/usr/bin/env python3
"""Startup rejects a truncated, reordered, or edited paper journal.

The hash chain covers new runtime lines only. No signing key is read.
Stdout is one JSON object. Research and simulated paper only.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_WORK = Path("/tmp/dipcatcher-ops17-tamper")


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def _drop_last(path: Path) -> None:
    rows = _lines(path)
    path.write_text("".join(rows[:-1]), encoding="utf-8")


def _swap(path: Path) -> None:
    rows = _lines(path)
    rows[0], rows[1] = rows[1], rows[0]
    path.write_text("".join(rows), encoding="utf-8")


def _edit(path: Path, old: str, new: str) -> None:
    path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")


def _partial(path: Path) -> None:
    with path.open("ab") as handle:
        handle.write(b'{"kind":')


def _case(source: Path, name: str, mutate) -> dict[str, object]:  # type: ignore[no-untyped-def]
    from quant_fund.api.lifecycle import acceptance_block, reset_runtime_state
    from quant_fund.observe.metrics import render_prometheus
    from quant_fund.paper.durable_book import file_digest, ledger_path, run_orders
    from quant_fund.paper.order_lease import release_order_lease
    from quant_fund.paper.runtime_log import AuditTamperError

    dest = _WORK / name
    shutil.copytree(source, dest)
    before = file_digest(ledger_path(dest))
    mutate(dest)
    code = ""
    try:
        run_orders(dest, 1, "halt")
    except AuditTamperError as exc:
        code = exc.code
    after = file_digest(ledger_path(dest))
    acceptance = acceptance_block()
    gauge = "dipcatcher_audit_tamper 1" in render_prometheus()
    same = before == after
    ok = bool(code) and same and gauge and acceptance == f"audit_{code}"
    release_order_lease(dest)
    reset_runtime_state()
    return {
        "acceptance": acceptance,
        "code": code,
        "digest_same": same,
        "gauge": gauge,
        "ok": ok,
    }


def execute() -> dict[str, object]:
    from quant_fund.api.lifecycle import reset_runtime_state
    from quant_fund.paper.durable_book import journal_path, open_book, submit_once
    from quant_fund.paper.order_lease import release_order_lease
    from quant_fund.paper.runtime_log import verify_state

    if _WORK.exists():
        shutil.rmtree(_WORK)
    source = _WORK / "clean"
    source.mkdir(parents=True)
    session = open_book(source)
    first = submit_once(session, source, "a")
    second = submit_once(session, source, "b")
    clean = verify_state(source) is None and first == "accepted" and second == "accepted"
    journal = journal_path(source)
    audit = source / "audit" / "audit.jsonl"
    truncated = _case(source, "truncated", lambda dest: _drop_last(journal_path(dest)))
    reordered = _case(source, "reordered", lambda dest: _swap(journal_path(dest)))
    edited = _case(
        source, "edited", lambda dest: _edit(journal_path(dest), '"side": "BUY"', '"side": "SELL"')
    )
    audit_edited = _case(
        source,
        "audit",
        lambda dest: _edit(
            dest / "audit" / "audit.jsonl", '"result": "accepted"', '"result": "changed"'
        ),
    )
    partial = _case(source, "partial", lambda dest: _partial(journal_path(dest)))
    release_order_lease(source)
    reset_runtime_state()
    ok = bool(
        clean
        and truncated["code"] == "truncated"
        and reordered["code"] == "reordered"
        and edited["code"] == "edited"
        and audit_edited["code"] == "edited"
        and partial["code"] == "truncated"
        and truncated["ok"]
        and reordered["ok"]
        and edited["ok"]
        and audit_edited["ok"]
        and partial["ok"]
    )
    return {
        "audit_edited": audit_edited,
        "clean": clean,
        "edited": edited,
        "error": "" if ok else "tamper",
        "journal_bytes": journal.stat().st_size,
        "ok": ok,
        "partial": partial,
        "reordered": reordered,
        "truncated": truncated,
        "audit_bytes": audit.stat().st_size,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-audit-tamper-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
