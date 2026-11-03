"""Read-only page, paper shutdown, backup restore, and soak bounds."""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
from pathlib import Path

import pytest
import yaml

from quant_fund.paper.durable_book import (
    append_json_line,
    content_digests,
    journal_path,
    ledger_order_count,
    ledger_path,
    open_book,
    pending_order_ids,
    read_json_lines,
    reconcile,
    request_stop,
    reset_stop,
    run_orders,
    submit_once,
)

_ROOT = Path(__file__).resolve().parents[3]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_read_only_page_ignores_tmpfs_and_leaves_disk_pressure() -> None:
    alerts = (_ROOT / "deploy" / "observability" / "alerts.yml").read_text(encoding="utf-8")
    case = yaml.safe_load(
        (_ROOT / "deploy" / "observability" / "fs_readonly_test.yml").read_text(encoding="utf-8")
    )
    assert "and node_filesystem_readonly == 0" in alerts
    assert 'fstype!~"tmpfs|devtmpfs|overlay|squashfs|ramfs|aufs"} == 1' in alerts
    document = yaml.safe_load(alerts)
    notes = {}
    for group in document["groups"]:
        for rule in group["rules"]:
            if rule.get("alert") == "DipcatcherFilesystemReadOnly":
                notes = rule["annotations"]
    assert notes["runbook"] == "docs/incidents/DipcatcherFilesystemReadOnly.md"
    page = (_ROOT / "docs" / "incidents" / "DipcatcherFilesystemReadOnly.md").read_text(
        encoding="utf-8"
    )
    for heading in (
        "## Trigger",
        "## Impact",
        "## First 5 minutes",
        "## Rollback and kill switch",
    ):
        assert heading in page
    fired = case["tests"][0]["alert_rule_test"][0]
    assert fired["exp_alerts"][0]["exp_annotations"] == notes
    assert case["tests"][0]["alert_rule_test"][1]["exp_alerts"] == []
    prod = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    root = (_ROOT / "Makefile").read_text(encoding="utf-8")
    for name in (
        "prod-fs-readonly",
        "prod-shutdown-drill",
        "prod-backup-restore-drill",
        "prod-paper-soak",
    ):
        assert f"{name}:" in prod
        assert f"{name}:" not in root


def test_fault_metrics_page_a_read_only_mount_and_healthy_clears_it() -> None:
    module = _load(_ROOT / "scripts" / "prod_alert_drill.py", "dipcatcher_prod_alert_drill_ops16")
    fault = module.fault_metrics(4)
    healthy = module.healthy_metrics(1_700_000_000, 200_000)
    assert 'node_filesystem_readonly{device="/dev/drill"' in fault
    assert 'node_filesystem_readonly{device="/dev/ro",fstype="ext4",mountpoint="/ro"} 1' in fault
    assert 'node_filesystem_readonly{device="/dev/ro",fstype="ext4",mountpoint="/ro"} 0' in healthy
    assert 'node_filesystem_readonly{device="/dev/drill"' in fault
    assert fault.count('node_filesystem_readonly{device="/dev/drill"') == 1


def test_intent_is_applied_once_and_a_duplicate_does_not_submit(tmp_path: Path) -> None:
    session = open_book(tmp_path)
    append_json_line(
        journal_path(tmp_path),
        {
            "kind": "intent",
            "order_id": "a",
            "quantity": 1,
            "schema": "paper_order_journal.v1",
            "security_id": "SYN",
            "side": "BUY",
        },
    )
    assert pending_order_ids(tmp_path) == ["a"]
    assert reconcile(session, tmp_path) == ["a"]
    digest = content_digests(tmp_path)["broker_state.json"]
    assert reconcile(session, tmp_path) == []
    assert content_digests(tmp_path)["broker_state.json"] == digest
    assert ledger_order_count(ledger_path(tmp_path), "a") == 1
    resumed = open_book(tmp_path)
    assert submit_once(resumed, tmp_path, "a") == "duplicate"
    assert ledger_order_count(ledger_path(tmp_path), "a") == 1
    assert content_digests(tmp_path)["broker_state.json"] == digest


def test_torn_journal_tail_is_ignored(tmp_path: Path) -> None:
    path = journal_path(tmp_path)
    path.parent.mkdir(parents=True)
    line = json.dumps(
        {"kind": "commit", "order_id": "a", "schema": "paper_order_journal.v1"},
        sort_keys=True,
    )
    path.write_text(line + '\n{"kind":', encoding="utf-8")
    rows = read_json_lines(path)
    assert len(rows) == 1
    assert rows[0]["order_id"] == "a"
    path.write_text(line + '\n{"kind":\n' + line + "\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_json_lines(path)


def test_stop_flag_finishes_the_current_order(tmp_path: Path) -> None:
    calls = {"n": 0}
    real = submit_once

    def _wrapped(session, state, order_id):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            request_stop(signal.SIGTERM, None)
        return real(session, state, order_id)

    module = sys.modules["quant_fund.paper.durable_book"]
    original = module.submit_once
    module.submit_once = _wrapped
    try:
        assert run_orders(tmp_path, 100, "t") == 0
        assert calls["n"] == 1
        assert ledger_order_count(ledger_path(tmp_path), "t-0") == 1
        assert pending_order_ids(tmp_path) == []
    finally:
        module.submit_once = original
        reset_stop()


def test_soak_bounds_flag_a_thread_climb_and_accept_a_flat_series() -> None:
    module = _load(_ROOT / "scripts" / "prod_paper_soak.py", "dipcatcher_prod_paper_soak_ops16")
    flat = [
        {"t": 0.0, "fds": 10.0, "rss_kb": 1000.0, "threads": 1.0},
        {"t": 130.0, "fds": 10.0, "rss_kb": 1000.0, "threads": 1.0},
        {"t": 160.0, "fds": 11.0, "rss_kb": 1100.0, "threads": 2.0},
        {"t": 190.0, "fds": 10.0, "rss_kb": 1000.0, "threads": 1.0},
    ]
    assert module.leak_bounds(flat)["leak"] is False
    climbing = [
        {"t": 0.0, "fds": 10.0, "rss_kb": 1000.0, "threads": 1.0},
        {"t": 130.0, "fds": 10.0, "rss_kb": 1000.0, "threads": 1.0},
        {"t": 160.0, "fds": 10.0, "rss_kb": 1000.0, "threads": 1.0},
        {"t": 190.0, "fds": 10.0, "rss_kb": 1000.0, "threads": 4.0},
    ]
    assert module.leak_bounds(climbing)["leak"] is True
    assert module._DEFAULT_SECONDS == 3600.0
