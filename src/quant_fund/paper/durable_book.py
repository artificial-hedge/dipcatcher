"""Paper order loop that journals intent before the ledger replace.

Simulated paper only. A SIGTERM or SIGINT finishes the order that is already
inside ``submit_once`` and then stops. A kill during the ledger replace leaves
the previous file in place; the intent line is how restart applies that order
once. Broker probes live in ``quant_fund.paper.dependencies`` and run
only when ``DIPCATCHER_PAPER_BROKER_URL`` is set. This module does not
open a broker socket itself.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import signal
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

from quant_fund.observe.order_guard import audit_tamper
from quant_fund.observe.write_halt import DurableWriteError, errno_code, note_write_failure
from quant_fund.paper.dependencies import dependency_refusal
from quant_fund.paper.order_lease import acquire_order_lease, assert_fence
from quant_fund.paper.runtime_log import AuditTamperError, append_runtime, verify_book
from quant_fund.paper.trading_session import PaperTradingSession, SessionCommandError

SCHEMA = "paper_order_journal.v1"
SECURITY = "SYN"
PRICE = 100.0
_STOP = False


def request_stop(_signum: int, _frame: object) -> None:
    """Latch a graceful stop. The current order still checkpoints."""
    global _STOP
    _STOP = True


def reset_stop() -> None:
    """Clear the stop latch. Tests and a new process start from here."""
    global _STOP
    _STOP = False


def stop_requested() -> bool:
    return _STOP


def ledger_path(state: Path) -> Path:
    return state / "broker_state.json"


def journal_path(state: Path) -> Path:
    return state / "journal" / "events.jsonl"


def audit_path(state: Path) -> Path:
    return state / "audit" / "audit.jsonl"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_digests(state: Path) -> dict[str, str]:
    """sha256 of the ledger, the journal, and the audit log."""
    found: dict[str, str] = {}
    for relative in ("broker_state.json", "journal/events.jsonl", "audit/audit.jsonl"):
        found[relative] = file_digest(state / relative)
    return found


def ledger_intact(path: Path) -> bool:
    """True when the published ledger is a complete paper checkpoint."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("schema") == "paper_broker_state.v1"


def ledger_order_count(path: Path, order_id: str) -> int:
    """How many times ``order_id`` appears in the published ledger."""
    if not ledger_intact(path):
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    orders = payload.get("orders")
    if not isinstance(orders, list):
        return 0
    count = 0
    for row in orders:
        if isinstance(row, dict) and row.get("order_id") == order_id:
            count += 1
    return count


def _refuse(exc: OSError) -> NoReturn:
    code = errno_code(exc.errno or 0)
    note_write_failure(code)
    raise DurableWriteError(code) from exc


def append_json_line(path: Path, row: Mapping[str, object]) -> None:
    """Append one JSON line and fsync it. A short write raises."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        _refuse(exc)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _refuse(OSError(errno.EIO, "short write"))
            view = view[written:]
        os.fsync(descriptor)
    except DurableWriteError:
        raise
    except OSError as exc:
        _refuse(exc)
    finally:
        os.close(descriptor)


def read_json_lines(path: Path) -> list[dict[str, object]]:
    """JSON lines. A torn final line is ignored. A torn middle line raises."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _order_id(row: Mapping[str, object]) -> str | None:
    if row.get("schema") != SCHEMA:
        return None
    oid = row.get("order_id")
    if isinstance(oid, str) and oid:
        return oid
    return None


def _ids(rows: list[dict[str, object]], kinds: set[str]) -> set[str]:
    found: set[str] = set()
    for row in rows:
        if row.get("kind") not in kinds:
            continue
        oid = _order_id(row)
        if oid is not None:
            found.add(oid)
    return found


def pending_order_ids(state: Path) -> list[str]:
    """Intent lines that have no commit or reject yet."""
    rows = read_json_lines(journal_path(state))
    closed = _ids(rows, {"commit", "reject"})
    pending: list[str] = []
    for row in rows:
        if row.get("kind") != "intent":
            continue
        oid = _order_id(row)
        if oid is None or oid in closed or oid in pending:
            continue
        pending.append(oid)
    return pending


def _ensure_files(state: Path) -> None:
    for path in (journal_path(state), audit_path(state)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"")


def open_book(state: Path) -> PaperTradingSession:
    """Resume a book, or mark SYN and checkpoint a new simulated book."""
    state.mkdir(parents=True, exist_ok=True)
    _ensure_files(state)
    path = ledger_path(state)
    if path.is_file():
        return PaperTradingSession.open(path, resume=True, starting_cash=1_000_000.0)
    session = PaperTradingSession.open(path, resume=False, starting_cash=1_000_000.0)
    session.apply_command({"op": "mark", "security_id": SECURITY, "price": PRICE})
    return session


def _ledger_ids(session: PaperTradingSession) -> set[str]:
    report = session.reconcile_view()
    orders = report.get("orders")
    found: set[str] = set()
    if not isinstance(orders, list):
        return found
    for row in orders:
        if not isinstance(row, dict):
            continue
        oid = row.get("order_id")
        if isinstance(oid, str):
            found.add(oid)
    return found


def _checkpoint_order(session: PaperTradingSession, order_id: str) -> str:
    step = session.apply_command(
        {
            "op": "submit",
            "order_id": order_id,
            "quantity": 1,
            "security_id": SECURITY,
            "side": "BUY",
        }
    )
    verdict = step.get("verdict")
    if isinstance(verdict, str):
        return verdict
    return "error"


def _startup(state: Path) -> None:
    """Hold the lease and refuse a tampered journal before any order."""
    acquire_order_lease(state)
    assert_fence(state)
    verify_book(state)


def _latched_tamper() -> None:
    code = audit_tamper()
    if code is not None:
        raise AuditTamperError(code)


def _close(state: Path, order_id: str, kind: str) -> None:
    append_runtime(
        state,
        "journal",
        {"kind": kind, "order_id": order_id, "schema": SCHEMA},
    )


def _audit(state: Path, order_id: str, result: str) -> None:
    append_runtime(
        state,
        "audit",
        {
            "event": "paper_order",
            "order_id": order_id,
            "research_only": True,
            "result": result,
            "schema": "paper_audit.v1",
        },
    )


def _intent(state: Path, order_id: str) -> None:
    append_runtime(
        state,
        "journal",
        {
            "kind": "intent",
            "order_id": order_id,
            "quantity": 1,
            "schema": SCHEMA,
            "security_id": SECURITY,
            "side": "BUY",
        },
    )


def reconcile(session: PaperTradingSession, state: Path) -> list[str]:
    """Apply each open intent that is not already in the ledger. Once."""
    _startup(state)
    if dependency_refusal() is not None:
        return []
    rows = read_json_lines(journal_path(state))
    closed = _ids(rows, {"commit", "reject"})
    present = _ledger_ids(session)
    applied: list[str] = []
    for row in rows:
        if row.get("kind") != "intent":
            continue
        oid = _order_id(row)
        if oid is None or oid in closed:
            continue
        closed.add(oid)
        if oid in present:
            _close(state, oid, "commit")
            continue
        verdict = _checkpoint_order(session, oid)
        if verdict != "accepted":
            raise SessionCommandError(verdict)
        _close(state, oid, "commit")
        _audit(state, oid, "reconciled")
        present.add(oid)
        applied.append(oid)
    return applied


def _already(session: PaperTradingSession, state: Path, order_id: str) -> bool:
    if order_id in _ledger_ids(session):
        return True
    rows = read_json_lines(journal_path(state))
    return order_id in _ids(rows, {"commit"})


def submit_once(session: PaperTradingSession, state: Path, order_id: str) -> str:
    """Journal the intent, checkpoint, then commit. A duplicate does not submit."""
    acquire_order_lease(state)
    assert_fence(state)
    _latched_tamper()
    blocked = dependency_refusal()
    if blocked is not None:
        return blocked
    if _already(session, state, order_id):
        return "duplicate"
    _intent(state, order_id)
    verdict = _checkpoint_order(session, order_id)
    if verdict != "accepted":
        _close(state, order_id, "reject")
        return verdict
    _close(state, order_id, "commit")
    _audit(state, order_id, "accepted")
    return "accepted"


def run_orders(state: Path, count: int, prefix: str) -> int:
    """Submit ``count`` new orders. A stop latch ends the loop between orders."""
    state.mkdir(parents=True, exist_ok=True)
    _startup(state)
    if dependency_refusal() is not None:
        return 1
    session = open_book(state)
    reconcile(session, state)
    if dependency_refusal() is not None:
        return 1
    (state / "ready").write_text("1\n", encoding="utf-8")
    used = _ids(read_json_lines(journal_path(state)), {"intent", "commit", "reject"})
    issued = 0
    seq = 0
    while issued < count:
        if stop_requested():
            break
        order_id = f"{prefix}-{seq}"
        seq += 1
        if order_id in used:
            continue
        verdict = submit_once(session, state, order_id)
        used.add(order_id)
        if verdict != "accepted":
            return 1
        issued += 1
    return 0


def serve(state: Path, count: int, prefix: str) -> int:
    """Install SIGTERM and SIGINT, run the loop, and restore the handlers."""
    reset_stop()
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        return run_orders(state, count, prefix)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        reset_stop()
