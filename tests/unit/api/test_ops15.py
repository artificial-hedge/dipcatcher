"""Config carry, a failed durable write, and a wall-clock step."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest import mock

import pytest
import yaml

from quant_fund.api.json_logging import append_audit_line
from quant_fund.api.lifecycle import acceptance_block, begin_drain, reset_runtime_state
from quant_fund.api.ops_metrics import sample_operator_gauges
from quant_fund.api.rate_limit import BUCKET, reset_bucket
from quant_fund.live.carry import content_digests, downgrade, upgrade
from quant_fund.live.clock_guard import STALE_SECONDS, ClockGuard, freshness_seconds
from quant_fund.live.config import LiveConfig
from quant_fund.live.session import open_session
from quant_fund.observe.metrics import render_prometheus
from quant_fund.observe.write_halt import (
    DurableWriteError,
    clear_write_halt,
    ensure_directories,
    note_write_failure,
    probe_directory,
    replace_text,
    write_halt_reason,
)
from quant_fund.paper.trading_session import PaperTradingSession
from quant_fund.utils.operator_kill import (
    ENV_PATH,
    HALT_NEW_ORDERS,
    orders_blocked,
    write_operator_kill,
)

_ROOT = Path(__file__).resolve().parents[3]
_PRETRADE = _ROOT / "configs" / "pretrade_risk.yaml"
_DAY_NS = 86_400 * 1_000_000_000
_FILES = (
    "ledger/broker_state.json",
    "journal/events.jsonl",
    "rate/window.json",
    "audit/audit.jsonl",
)


def _tree(root: Path, config_schema: int) -> None:
    payloads = {
        "ledger/broker_state.json": '{"schema":"paper_broker_state.v1"}\n',
        "journal/events.jsonl": '{"data_label":"SYNTHETIC"}\n',
        "rate/window.json": '{"order_ts":[1]}\n',
        "audit/audit.jsonl": '{"event":"seed"}\n',
    }
    for relative, text in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    body = {"config_schema": config_schema, "order_path": "paper", "state_schema": 1}
    if config_schema == 2:
        body["rate_limit_rps"] = 50
    (root / "config.json").write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")


def _paper(root: Path) -> PaperTradingSession:
    path = root / "broker_state.json"
    book = PaperTradingSession.open(
        path, resume=False, starting_cash=1_000_000.0, endpoint_kind="paper"
    )
    marked = book.apply_command({"op": "mark", "security_id": "AAPL", "price": 100.0})
    assert marked["verdict"] == "marked"
    return book


def _submit(book: PaperTradingSession, order_id: str) -> dict[str, object]:
    return book.apply_command(
        {
            "op": "submit",
            "order_id": order_id,
            "security_id": "AAPL",
            "side": "BUY",
            "quantity": 1.0,
        }
    )


def test_upgrade_and_downgrade_keep_state_bytes(tmp_path: Path) -> None:
    _tree(tmp_path, 1)
    before = content_digests(tmp_path)
    assert upgrade(tmp_path)["ok"] is True
    mid = content_digests(tmp_path)
    assert downgrade(tmp_path)["ok"] is True
    after = content_digests(tmp_path)
    assert before == mid == after
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["config_schema"] == 1
    assert "rate_limit_rps" not in config
    for relative in _FILES:
        assert len(before[relative]) == 64


def test_newer_state_schema_fails_closed(tmp_path: Path) -> None:
    _tree(tmp_path, 1)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["state_schema"] = 99
    (tmp_path / "config.json").write_text(
        json.dumps(config, sort_keys=True) + "\n", encoding="utf-8"
    )
    before = {name: (tmp_path / name).read_bytes() for name in _FILES}
    report = downgrade(tmp_path)
    assert report["ok"] is False
    assert report["error"] == (
        "state schema 99 is newer than this process; refusing to rewrite state"
    )
    assert content_digests(tmp_path) == report["digests"]
    for name, payload in before.items():
        assert (tmp_path / name).read_bytes() == payload


def test_newer_config_schema_fails_closed(tmp_path: Path) -> None:
    _tree(tmp_path, 1)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["config_schema"] = 3
    (tmp_path / "config.json").write_text(
        json.dumps(config, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = upgrade(tmp_path)
    assert report["ok"] is False
    assert "config schema 3 is newer than this process" in str(report["error"])


def test_clock_guard_freezes_the_rate_timestamp() -> None:
    guard = ClockGuard()
    assert guard.order_ns(100, 0) == 100
    assert guard.order_ns(150, 50) == 150
    stepped = guard.order_ns(150 + _DAY_NS, 51)
    assert guard.forward
    assert stepped == 151


def test_clock_guard_does_not_open_a_fresh_window() -> None:
    guard = ClockGuard()
    assert guard.order_ns(1000, 0) == 1000
    assert guard.order_ns(1000 - _DAY_NS, 10) == 1010
    assert guard.backward
    assert guard.order_ns(1000 - _DAY_NS, 20) == 1020


def test_backward_step_is_stale() -> None:
    assert freshness_seconds(10.0, 20.0, backward=False) == STALE_SECONDS
    assert freshness_seconds(50.0, 10.0, backward=True) == STALE_SECONDS
    assert freshness_seconds(50.0, 10.0, backward=False) == 40.0


def test_read_only_probe_latches_and_then_clears(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory mode bits")
    clear_write_halt()
    os.chmod(tmp_path, 0o555)
    try:
        assert probe_directory(tmp_path) == "read_only"
        assert ensure_directories([tmp_path]) == "read_only"
        assert write_halt_reason() == "read_only"
    finally:
        os.chmod(tmp_path, 0o755)
        clear_write_halt()
    assert ensure_directories([tmp_path]) is None
    assert write_halt_reason() is None
    clear_write_halt()


def test_replace_text_keeps_the_previous_bytes(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory mode bits")
    target = tmp_path / "broker_state.json"
    target.write_text("old\n", encoding="utf-8")
    os.chmod(tmp_path, 0o555)
    try:
        with pytest.raises(DurableWriteError) as caught:
            replace_text(target, "new\n")
        assert caught.value.code == "read_only"
        assert target.read_text(encoding="utf-8") == "old\n"
        assert write_halt_reason() == "read_only"
    finally:
        os.chmod(tmp_path, 0o755)
        clear_write_halt()


def test_http_bucket_ignores_a_wall_jump(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"mono": 1_000.0}
    monkeypatch.setenv("DIPCATCHER_RATE_LIMIT_RPS", "1")
    monkeypatch.delenv("DIPCATCHER_PROFILE", raising=False)
    reset_bucket()
    try:
        with mock.patch("quant_fund.api.rate_limit.time.monotonic", lambda: clock["mono"]):
            first, _retry = BUCKET.take()
            second, _again = BUCKET.take()
            with mock.patch("quant_fund.api.rate_limit.time.time", lambda: 9_000_000_000.0):
                third, _later = BUCKET.take()
        assert first
        assert not second
        assert not third
    finally:
        reset_bucket()


def test_kill_file_is_not_cleared_by_a_jump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DIPCATCHER_PROFILE", raising=False)
    monkeypatch.delenv("DIPCATCHER_WRITE_OWNER", raising=False)
    monkeypatch.delenv(ENV_PATH, raising=False)
    path = tmp_path / "kill_switch"
    write_operator_kill(path, HALT_NEW_ORDERS)
    guard = ClockGuard()
    guard.order_ns(100, 0)
    guard.order_ns(100 + _DAY_NS, 10)
    guard.order_ns(100 - _DAY_NS, 20)
    assert orders_blocked(path) == f"kill_switch:{HALT_NEW_ORDERS}"


def test_paper_submit_halts_on_a_read_only_ledger(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory mode bits")
    clear_write_halt()
    book = _paper(tmp_path)
    path = tmp_path / "broker_state.json"
    before = path.read_bytes()
    os.chmod(tmp_path, 0o555)
    try:
        refused = _submit(book, "ro-1")
        assert refused["verdict"] == "rejected"
        assert refused["reason"] == "durable_read_only"
        assert path.read_bytes() == before
        assert write_halt_reason() == "read_only"
    finally:
        os.chmod(tmp_path, 0o755)
        clear_write_halt()
    accepted = _submit(book, "rw-1")
    assert accepted["verdict"] == "accepted"
    assert write_halt_reason() is None
    clear_write_halt()


def test_checkpoint_failure_drops_the_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_write_halt()
    book = _paper(tmp_path)
    path = tmp_path / "broker_state.json"
    before = path.read_bytes()

    def boom(_path: Path, _payload: str) -> None:
        note_write_failure("enospc")
        raise DurableWriteError("enospc")

    monkeypatch.setattr("quant_fund.paper.broker_adapter.replace_text", boom)
    try:
        refused = _submit(book, "nope")
        assert refused["verdict"] == "rejected"
        assert refused["reason"] == "durable_enospc"
        assert path.read_bytes() == before
        rows = book.reconcile_view()["orders"]
        assert "nope" not in [row["order_id"] for row in rows]
        assert write_halt_reason() == "enospc"
    finally:
        clear_write_halt()


def test_audit_append_does_not_drop_a_failed_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory mode bits")
    path = tmp_path / "audit.jsonl"
    path.write_text("kept\n", encoding="utf-8")
    path.chmod(0o444)
    monkeypatch.setenv("DIPCATCHER_AUDIT_LOG_PATH", str(path))
    clear_write_halt()
    try:
        with pytest.raises(DurableWriteError) as caught:
            append_audit_line("dropped")
        assert caught.value.code == "read_only"
        assert path.read_text(encoding="utf-8") == "kept\n"
        assert write_halt_reason() == "read_only"
        link = tmp_path / "linked.jsonl"
        link.symlink_to(path)
        monkeypatch.setenv("DIPCATCHER_AUDIT_LOG_PATH", str(link))
        append_audit_line("ignored")
        assert path.read_text(encoding="utf-8") == "kept\n"
    finally:
        path.chmod(0o644)
        clear_write_halt()


def test_acceptance_block_reports_the_latch_under_the_drain() -> None:
    reset_runtime_state()
    clear_write_halt()
    try:
        assert acceptance_block() is None
        note_write_failure("enospc")
        assert acceptance_block() == "durable_enospc"
        ready, _draining, blocking, code = sample_operator_gauges()
        assert ready == 0.0
        assert blocking == 1.0
        assert code == "durable_enospc"
        begin_drain()
        assert acceptance_block() == "draining"
        text = render_prometheus()
        assert "dipcatcher_write_halt 1" in text
    finally:
        reset_runtime_state()
        clear_write_halt()
    assert acceptance_block() is None
    assert "dipcatcher_write_halt" not in render_prometheus()


def test_session_clock_uses_the_guard(tmp_path: Path) -> None:
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "loopback",
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(_PRETRADE),
            "state_dir": str(tmp_path),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )
    session = open_session(config, _PRETRADE, state_dir=tmp_path, now_ns=lambda: 0)
    walls = iter([1_000, 1_050, 1_050 + _DAY_NS])
    monos = iter([0, 50, 51])
    session._now = lambda: next(walls)
    with mock.patch("quant_fund.live.session.time.monotonic_ns", lambda: next(monos)):
        assert session._clock(None) == 1_000
        assert session._clock(None) == 1_050
        assert session._clock(None) == 1_051
    assert session._guard.forward
    assert session._clock(5) == 5


def test_slo_promtool_expects_the_runbook_annotation() -> None:
    rules = yaml.safe_load(
        (_ROOT / "deploy" / "observability" / "slo_rules.yml").read_text(encoding="utf-8")
    )
    case = yaml.safe_load(
        (_ROOT / "deploy" / "observability" / "slo_rules_test.yml").read_text(encoding="utf-8")
    )
    expected: dict[str, dict[str, str]] = {}
    for group in rules["groups"]:
        for rule in group["rules"]:
            name = rule.get("alert")
            if name:
                expected[str(name)] = {
                    str(key): str(value) for key, value in rule["annotations"].items()
                }
    seen: set[str] = set()
    for block in case["tests"]:
        for check in block["alert_rule_test"]:
            alerts = check["exp_alerts"]
            if not alerts:
                continue
            name = str(check["alertname"])
            seen.add(name)
            assert alerts[0]["exp_annotations"] == expected[name]
            assert expected[name]["runbook"] == f"docs/incidents/{name}.md"
    assert seen == set(expected)


def test_targets_live_in_prod_mk() -> None:
    prod = (_ROOT / "deploy" / "prod.mk").read_text(encoding="utf-8")
    root = (_ROOT / "Makefile").read_text(encoding="utf-8")
    for name in ("prod-migrate-drill", "prod-disk-drill", "prod-time-drill"):
        assert f"{name}:" in prod
        assert f"{name}:" not in root
    page = (_ROOT / "docs" / "incidents" / "DipcatcherWriteHalt.md").read_text(encoding="utf-8")
    for heading in (
        "## Trigger",
        "## Impact",
        "## First 5 minutes",
        "## Rollback and kill switch",
    ):
        assert heading in page
    alerts = (_ROOT / "deploy" / "observability" / "alerts.yml").read_text(encoding="utf-8")
    assert "DipcatcherWriteHalt" in alerts
    assert "dipcatcher_write_halt == 1" in alerts


def test_digest_helper_is_sha256(tmp_path: Path) -> None:
    _tree(tmp_path, 2)
    found = content_digests(tmp_path)
    raw = (tmp_path / "journal/events.jsonl").read_bytes()
    assert found["journal/events.jsonl"] == hashlib.sha256(raw).hexdigest()
