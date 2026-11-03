"""Paper lease, dependency probes, journal chain, and log rotation."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from quant_fund.api.lifecycle import acceptance_block, reset_runtime_state
from quant_fund.observe.metrics import render_prometheus
from quant_fund.paper.durable_book import (
    journal_path,
    ledger_order_count,
    ledger_path,
    open_book,
    read_json_lines,
    run_orders,
    submit_once,
)
from quant_fund.paper.order_lease import (
    SplitBrainError,
    acquire_order_lease,
    lock_path,
    read_disk_token,
    release_order_lease,
)
from quant_fund.paper.runtime_log import AuditTamperError, verify_state

_ROOT = Path(__file__).resolve().parents[3]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802
        delay = float(getattr(self.server, "delay", 0.0))
        if delay:
            time.sleep(delay)
        body = b"ok"
        try:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _serve(delay: float) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.delay = delay  # type: ignore[attr-defined]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _intents(state: Path, order_id: str) -> int:
    path = journal_path(state)
    if not path.is_file():
        return 0
    rows = read_json_lines(path)
    return sum(1 for row in rows if row.get("kind") == "intent" and row.get("order_id") == order_id)


def test_stale_lock_is_incremented_on_the_same_inode(tmp_path: Path) -> None:
    path = lock_path(tmp_path)
    path.write_text(
        json.dumps({"pid": 2**20, "schema": "paper_order_lease.v1", "token": 4}, sort_keys=True),
        encoding="utf-8",
    )
    inode = path.stat().st_ino
    try:
        assert acquire_order_lease(tmp_path) == 5
        assert path.stat().st_ino == inode
        assert path.is_file()
        assert read_disk_token(tmp_path) == 5
        assert acquire_order_lease(tmp_path) == 5
    finally:
        release_order_lease(tmp_path)
    assert path.is_file()
    assert acquire_order_lease(tmp_path) == 6
    release_order_lease(tmp_path)


def test_a_rewritten_token_refuses_the_next_submit(tmp_path: Path) -> None:
    session = open_book(tmp_path)
    try:
        assert submit_once(session, tmp_path, "fence-1") == "accepted"
        lock_path(tmp_path).write_text(
            json.dumps({"pid": os.getpid(), "schema": "paper_order_lease.v1", "token": 99}),
            encoding="utf-8",
        )
        with pytest.raises(SplitBrainError):
            submit_once(session, tmp_path, "fence-2")
        assert ledger_order_count(ledger_path(tmp_path), "fence-1") == 1
        assert ledger_order_count(ledger_path(tmp_path), "fence-2") == 0
        assert acceptance_block() == "split_brain"
        assert "dipcatcher_split_brain 1" in render_prometheus()
    finally:
        release_order_lease(tmp_path)
        reset_runtime_state()


def test_second_process_submits_nothing(tmp_path: Path) -> None:
    script = _ROOT / "scripts" / "prod_split_brain_drill.py"
    holder = subprocess.Popen(
        [sys.executable, str(script), "hold", str(tmp_path)],
        cwd=_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    loser = None
    try:
        assert holder.stdout is not None
        line = holder.stdout.readline()
        assert json.loads(line)["token"] == 1
        loser = subprocess.Popen(
            [sys.executable, str(script), "submit", str(tmp_path), "loser"],
            cwd=_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert loser.wait(timeout=90) == 2
        assert loser.stdout is not None
        body = json.loads(loser.stdout.read())
        assert body["error"] == "split_brain"
        assert body["acceptance"] == "split_brain"
        assert "dipcatcher_split_brain 1" in body["metrics"]
        assert not ledger_path(tmp_path).exists()
    finally:
        if holder.poll() is None:
            os.kill(holder.pid, signal.SIGKILL)
        holder.wait(timeout=30)
        if loser is not None and loser.poll() is None:
            os.kill(loser.pid, signal.SIGKILL)
            loser.wait(timeout=30)


def _book(state: Path) -> None:
    session = open_book(state)
    assert submit_once(session, state, "a") == "accepted"
    assert submit_once(session, state, "b") == "accepted"
    assert verify_state(state) is None


def test_startup_halts_on_truncated_reordered_and_edited_lines(tmp_path: Path) -> None:
    _book(tmp_path)
    journal = journal_path(tmp_path)
    rows = journal.read_text(encoding="utf-8").splitlines(keepends=True)
    journal.write_text("".join(rows[:-1]), encoding="utf-8")
    try:
        with pytest.raises(AuditTamperError) as truncated:
            run_orders(tmp_path, 1, "halt")
        assert truncated.value.code == "truncated"
        assert acceptance_block() == "audit_truncated"
        assert "dipcatcher_audit_tamper 1" in render_prometheus()
        reset_runtime_state()
        text = journal.read_text(encoding="utf-8")
        swapped = "".join(reversed(text.splitlines(keepends=True)))
        journal.write_text(swapped, encoding="utf-8")
        # The tip still describes the original chain, and the links are backwards.
        with pytest.raises(AuditTamperError) as reordered:
            run_orders(tmp_path, 1, "halt")
        assert reordered.value.code == "reordered"
    finally:
        release_order_lease(tmp_path)
        reset_runtime_state()


def test_an_edited_audit_line_halts_and_a_legacy_prefix_still_submits(tmp_path: Path) -> None:
    state = tmp_path / "edit"
    _book(state)
    audit = state / "audit" / "audit.jsonl"
    audit.write_text(
        audit.read_text(encoding="utf-8").replace('"result": "accepted"', '"result": "changed"', 1),
        encoding="utf-8",
    )
    try:
        with pytest.raises(AuditTamperError) as edited:
            run_orders(state, 1, "halt")
        assert edited.value.code == "edited"
        assert ledger_order_count(ledger_path(state), "halt-0") == 0
    finally:
        release_order_lease(state)
        reset_runtime_state()
    legacy = tmp_path / "legacy"
    from quant_fund.paper.durable_book import append_json_line, reconcile

    session = open_book(legacy)
    append_json_line(
        journal_path(legacy),
        {
            "kind": "intent",
            "order_id": "old",
            "quantity": 1,
            "schema": "paper_order_journal.v1",
            "security_id": "SYN",
            "side": "BUY",
        },
    )
    try:
        assert reconcile(session, legacy) == ["old"]
        assert verify_state(legacy) is None
        assert submit_once(session, legacy, "old") == "duplicate"
    finally:
        release_order_lease(legacy)
        reset_runtime_state()


def test_a_torn_tail_stays_a_parser_skip_and_startup_calls_it_truncated(tmp_path: Path) -> None:
    path = journal_path(tmp_path)
    path.parent.mkdir(parents=True)
    line = json.dumps({"kind": "commit", "order_id": "a", "schema": "paper_order_journal.v1"})
    path.write_text(line + '\n{"kind":', encoding="utf-8")
    rows = read_json_lines(path)
    assert rows[0]["order_id"] == "a"
    assert verify_state(tmp_path) == "truncated"


def test_dependency_probes_fail_closed_and_recover_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    healthy = _serve(0.0)
    slow = _serve(1.5)
    refused = socket.socket()
    refused.bind(("127.0.0.1", 0))
    try:
        healthy_url = f"http://127.0.0.1:{healthy.server_address[1]}/"
        cases = [
            ("dns", "http://dipcatcher-ops17-nx.invalid:9/", "1", "dependency_dns"),
            (
                "down",
                f"http://127.0.0.1:{refused.getsockname()[1]}/",
                "1",
                "dependency_broker_down",
            ),
            (
                "slow",
                f"http://127.0.0.1:{slow.server_address[1]}/",
                "0.2",
                "dependency_broker_slow",
            ),
        ]
        for name, url, timeout, acceptance in cases:
            state = tmp_path / name
            monkeypatch.setenv("DIPCATCHER_PAPER_BROKER_URL", url)
            monkeypatch.setenv("DIPCATCHER_PAPER_BROKER_TIMEOUT", timeout)
            assert run_orders(state, 1, name) == 1
            assert _intents(state, f"{name}-0") == 0
            assert acceptance_block() == acceptance
            assert "dipcatcher_dependency_down 1" in render_prometheus()
            monkeypatch.setenv("DIPCATCHER_PAPER_BROKER_URL", healthy_url)
            monkeypatch.setenv("DIPCATCHER_PAPER_BROKER_TIMEOUT", "2")
            assert run_orders(state, 1, name) == 0
            assert submit_once(open_book(state), state, f"{name}-0") == "duplicate"
            assert ledger_order_count(ledger_path(state), f"{name}-0") == 1
            assert _intents(state, f"{name}-0") == 1
            assert "dipcatcher_dependency_down" not in render_prometheus()
            release_order_lease(state)
            reset_runtime_state()
    finally:
        refused.close()
        healthy.shutdown()
        slow.shutdown()
        release_order_lease()
        reset_runtime_state()


def test_expired_certificate_fails_the_tls_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drill = _load(
        _ROOT / "scripts" / "prod_dependency_drill.py", "dipcatcher_prod_dependency_ops17"
    )
    cert, key, ca, enddate = drill._expired(tmp_path)
    server = drill._https(cert, key)
    try:
        port = int(server.server_address[1])
        detail = drill._peek_tls(port, ca)
        assert "2020" in enddate
        assert "expired" in detail.lower()
        monkeypatch.setenv("DIPCATCHER_PAPER_BROKER_URL", f"https://127.0.0.1:{port}/")
        monkeypatch.setenv("DIPCATCHER_PAPER_BROKER_TIMEOUT", "2")
        monkeypatch.setenv("DIPCATCHER_PAPER_BROKER_CA", str(ca))
        state = tmp_path / "book"
        assert run_orders(state, 1, "tls") == 1
        assert _intents(state, "tls-0") == 0
        assert acceptance_block() == "dependency_tls"
        assert "dipcatcher_dependency_down 1" in render_prometheus()
    finally:
        server.shutdown()
        release_order_lease()
        reset_runtime_state()


def test_rotation_keeps_the_journal_and_the_order_path(tmp_path: Path) -> None:
    module = _load(_ROOT / "scripts" / "log_retain.py", "dipcatcher_log_retain_ops17")
    for stream in ("app", "proxy", "audit"):
        (tmp_path / stream).mkdir()
    service = tmp_path / "app" / "service.log"
    service.write_bytes(b"x" * 50)
    journal = tmp_path / "audit" / "events.jsonl"
    journal.write_bytes(b"j" * 80)
    before = journal.read_bytes()
    policy = module.Policy(10, 86400, 5)
    policies = {stream: policy for stream in ("app", "proxy", "audit")}
    report = module.retain_tree(tmp_path, policies, time.time())
    assert report["ok"] is True
    assert not service.exists()
    assert (tmp_path / "app" / "service.log.1").read_bytes() == b"x" * 50
    assert journal.read_bytes() == before
    assert module.sealed_name("events.jsonl")
    assert module.sealed_name("events.jsonl.2")
    state = tmp_path / "book"
    assert not lock_path(state).exists()
    try:
        assert submit_once(open_book(state), state, "rot-1") == "accepted"
        assert submit_once(open_book(state), state, "rot-1") == "duplicate"
    finally:
        release_order_lease(state)
        reset_runtime_state()


def test_guard_targets_and_fault_series_are_wired() -> None:
    prod = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    root = (_ROOT / "Makefile").read_text(encoding="utf-8")
    for name in (
        "prod-split-brain-drill",
        "prod-dependency-drill",
        "prod-audit-tamper-drill",
        "prod-log-journal-drill",
    ):
        assert f"{name}:" in prod
        assert f"{name}:" not in root
    alerts = (_ROOT / "deploy" / "observability" / "alerts.yml").read_text(encoding="utf-8")
    for name in ("DipcatcherSplitBrain", "DipcatcherDependencyDown", "DipcatcherAuditTamper"):
        assert name in alerts
    module = _load(_ROOT / "scripts" / "prod_alert_drill.py", "dipcatcher_prod_alert_drill_ops17")
    fault = module.fault_metrics(3)
    healthy = module.healthy_metrics(1_700_000_000, 10)
    for series in (
        "dipcatcher_split_brain",
        "dipcatcher_dependency_down",
        "dipcatcher_audit_tamper",
    ):
        assert f"{series} 1" in fault
        assert f"{series} 0" in healthy
    assert "dipcatcher_split_brain" not in render_prometheus()
