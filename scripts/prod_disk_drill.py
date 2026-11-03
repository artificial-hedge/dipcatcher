#!/usr/bin/env python3
"""Fill a tiny filesystem, then remount it read-only, and watch orders halt.

The ledger, journal, rate store, and audit file live on that mount. A failed
probe rejects the order, leaves the digests unchanged, and latches
``dipcatcher_write_halt``. Freeing space or remounting read-write clears the
latch. Stdout is one JSON object. This is simulated paper.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from quant_fund.observe.write_halt import clear_write_halt, write_halt_reason
from quant_fund.paper.trading_session import PaperTradingSession

_ROOT = Path(__file__).resolve().parents[1]
_ALERTS = _ROOT / "deploy" / "observability" / "alerts.yml"
_MOUNT = Path("/tmp/dipcatcher-ops15-disk")
_NAMES = (
    "ledger/broker_state.json",
    "journal/events.jsonl",
    "rate/window.json",
    "audit/audit.jsonl",
)


def _docker() -> list[str]:
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    return [*shlex.split(sudo), "docker"]


def _sudo() -> list[str]:
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    return shlex.split(sudo) if sudo else []


def _run(argv: list[str]) -> int:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    return proc.returncode


def _mount() -> bool:
    if _MOUNT.exists():
        _run([*_sudo(), "umount", str(_MOUNT)])
        shutil.rmtree(_MOUNT, ignore_errors=True)
    _MOUNT.mkdir(parents=True)
    uid = str(os.getuid())
    gid = str(os.getgid())
    return (
        _run(
            [
                *_sudo(),
                "mount",
                "-t",
                "tmpfs",
                "-o",
                f"size=256k,mode=0755,uid={uid},gid={gid}",
                "tmpfs",
                str(_MOUNT),
            ]
        )
        == 0
    )


def _umount() -> None:
    _run([*_sudo(), "umount", str(_MOUNT)])
    shutil.rmtree(_MOUNT, ignore_errors=True)


def _digests() -> dict[str, str]:
    found: dict[str, str] = {}
    for relative in _NAMES:
        path = _MOUNT / relative
        found[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    return found


def _seed() -> PaperTradingSession | None:
    for name in ("ledger", "journal", "rate", "audit"):
        (_MOUNT / name).mkdir()
    (_MOUNT / "journal" / "events.jsonl").write_text(
        '{"data_label":"SYNTHETIC","schema":"journal.v1"}\n', encoding="utf-8"
    )
    (_MOUNT / "rate" / "window.json").write_text('{"schema":"rate.v1"}\n', encoding="utf-8")
    (_MOUNT / "audit" / "audit.jsonl").write_text(
        '{"data_label":"SYNTHETIC","event":"seed"}\n', encoding="utf-8"
    )
    paper = _MOUNT / "ledger" / "broker_state.json"
    book = PaperTradingSession.open(
        paper, resume=False, starting_cash=1_000_000.0, endpoint_kind="paper"
    )
    marked = book.apply_command({"op": "mark", "security_id": "AAPL", "price": 100.0})
    accepted = book.apply_command(
        {
            "op": "submit",
            "order_id": "disk-1",
            "security_id": "AAPL",
            "side": "BUY",
            "quantity": 1.0,
        }
    )
    if marked.get("verdict") != "marked" or accepted.get("verdict") != "accepted":
        return None
    return book


def _env() -> dict[str, str]:
    return {
        "DIPCATCHER_LEDGER_PATH": str(_MOUNT / "ledger" / "broker_state.json"),
        "DIPCATCHER_JOURNAL_PATH": str(_MOUNT / "journal" / "events.jsonl"),
        "DIPCATCHER_RATE_STORE_PATH": str(_MOUNT / "rate" / "window.json"),
        "DIPCATCHER_AUDIT_LOG_PATH": str(_MOUNT / "audit" / "audit.jsonl"),
    }


def _submit(book: PaperTradingSession, order_id: str) -> str:
    verdict = book.apply_command(
        {
            "op": "submit",
            "order_id": order_id,
            "security_id": "AAPL",
            "side": "BUY",
            "quantity": 1.0,
        }
    )
    reason = verdict.get("reason")
    if verdict.get("verdict") == "accepted":
        return "accepted"
    return str(reason or verdict.get("verdict") or "")


def _fill() -> None:
    """Consume the mount, including the tail a one-byte probe would use."""
    target = _MOUNT / "filler"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        _write_until_full(fd, b"\0" * 4096)
        _write_until_full(fd, b"\0")
    finally:
        os.close(fd)


def _write_until_full(fd: int, blob: bytes) -> None:
    while True:
        try:
            written = os.write(fd, blob)
            os.fsync(fd)
        except OSError:
            return
        if written <= 0 or written < len(blob):
            return


def _promtool(work: Path) -> int:
    document = yaml.safe_load(_ALERTS.read_text(encoding="utf-8"))
    notes: dict[str, str] = {}
    for group in document["groups"]:
        for rule in group["rules"]:
            if rule.get("alert") == "DipcatcherWriteHalt":
                notes = {
                    str(key): str(value) for key, value in (rule.get("annotations") or {}).items()
                }
    if not notes:
        return 1
    case = {
        "rule_files": ["alerts.yml"],
        "evaluation_interval": "1m",
        "tests": [
            {
                "interval": "1m",
                "input_series": [{"series": "dipcatcher_write_halt", "values": "1+0x30"}],
                "alert_rule_test": [
                    {
                        "eval_time": "10m",
                        "alertname": "DipcatcherWriteHalt",
                        "exp_alerts": [
                            {"exp_labels": {"severity": "page"}, "exp_annotations": notes}
                        ],
                    }
                ],
            }
        ],
    }
    work.mkdir(parents=True, exist_ok=True)
    (work / "disk_drill.yml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    shutil.copyfile(_ALERTS, work / "alerts.yml")
    proc = subprocess.run(
        [
            *_docker(),
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "promtool",
            "-v",
            f"{work}:/rules:ro",
            "-w",
            "/rules",
            "prom/prometheus:v2.54.1",
            "test",
            "rules",
            "disk_drill.yml",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return proc.returncode


def execute() -> dict[str, object]:
    """Prove ENOSPC and a read-only mount halt orders without dropping bytes."""
    clear_write_halt()
    saved = {key: os.environ.get(key) for key in _env()}
    if not _mount():
        return {"error": "mount", "ok": False}
    try:
        for key, value in _env().items():
            os.environ[key] = value
        book = _seed()
        if book is None:
            return {"error": "seed", "ok": False}
        before = _digests()
        _fill()
        full = _submit(book, "disk-full")
        held = _digests() == before and write_halt_reason() == "enospc"
        filler = _MOUNT / "filler"
        if filler.exists():
            filler.unlink()
        clear_write_halt()
        recovered = _submit(book, "disk-recovered")
        after = _digests()
        readonly = _run([*_sudo(), "mount", "-o", "remount,ro", str(_MOUNT)]) == 0
        frozen = _submit(book, "disk-ro") if readonly else ""
        ro_held = readonly and _digests() == after and write_halt_reason() == "read_only"
        _run([*_sudo(), "mount", "-o", "remount,rw", str(_MOUNT)])
        clear_write_halt()
        again = _submit(book, "disk-rw")
        work = Path("/tmp/dipcatcher-ops15-disk-rules")
        if work.exists():
            shutil.rmtree(work)
        prom = _promtool(work)
        ok = (
            full == "durable_enospc"
            and held
            and recovered == "accepted"
            and frozen == "durable_read_only"
            and ro_held
            and again == "accepted"
            and prom == 0
        )
        return {
            "ledger_sha256": before.get("ledger/broker_state.json", ""),
            "ledger_sha256_after_recovery": after.get("ledger/broker_state.json", ""),
            "enospc": full,
            "error": "" if ok else "disk",
            "ok": ok,
            "promtool_exit": prom,
            "read_only": frozen,
            "recovered": recovered,
            "restored": again,
        }
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        clear_write_halt()
        _umount()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-disk-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
