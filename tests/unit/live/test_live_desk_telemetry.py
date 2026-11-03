"""Live-desk telemetry is bounded, deduplicated, and free of order details."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from quant_fund.live.bridge import BridgeError
from quant_fund.live.desk import LiveDeskError
from quant_fund.observe.logging import set_log_sink
from quant_fund.observe.metrics import render_prometheus, reset_metrics
from tests.unit.live.test_live_desk import _open
from tests.unit.pretrade.support import TS_NS

_LOG_KEYS = frozenset({"ts", "level", "event", "correlation_id", "service", "outcome", "venue"})
_FORBIDDEN_LOG_KEYS = frozenset(
    {
        "actor",
        "broker_reason",
        "client_order_id",
        "error",
        "event_id",
        "exception",
        "limit_price",
        "operator_reason",
        "order_id",
        "path",
        "payload",
        "price",
        "px",
        "qty",
        "quantity",
        "raw_exception",
        "reason",
        "reject_reason",
        "root",
        "root_path",
        "symbol",
    }
)
_VENUES = frozenset({"shadow", "loopback", "drop"})


@pytest.fixture
def telemetry(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    reset_metrics()
    sink: list[dict[str, Any]] = []
    set_log_sink(sink)
    try:
        yield sink
    finally:
        set_log_sink(None)
        reset_metrics()


def _event_count(logs: list[dict[str, Any]], event: str) -> int:
    return sum(record.get("event") == f"live_desk_{event}" for record in logs)


def _log_count(logs: list[dict[str, Any]], event: str, outcome: str, venue: str) -> int:
    return sum(
        record.get("event") == f"live_desk_{event}"
        and record.get("outcome") == outcome
        and record.get("venue") == venue
        for record in logs
    )


def _counter(event: str, outcome: str, venue: str) -> int:
    prefix = (
        f'dipcatcher_live_desk_events_total{{event="{event}",outcome="{outcome}",venue="{venue}"}} '
    )
    lines = [line for line in render_prometheus().splitlines() if line.startswith(prefix)]
    if not lines:
        return 0
    assert len(lines) == 1
    return int(lines[0].removeprefix(prefix))


def _assert_clean_logs(logs: list[dict[str, Any]], *canaries: str) -> None:
    serialized = json.dumps(logs, sort_keys=True, default=str)
    for canary in canaries:
        assert canary not in serialized
    for record in logs:
        assert set(record) == _LOG_KEYS
        assert _FORBIDDEN_LOG_KEYS.isdisjoint(record)
        assert str(record["event"]).startswith("live_desk_")
        assert record["venue"] in _VENUES


def _assert_no_submit_or_fill(logs: list[dict[str, Any]], venue: str) -> None:
    assert _event_count(logs, "order_submit") == 0
    assert _event_count(logs, "fill") == 0
    assert _counter("order_submit", "attempted", venue) == 0
    assert _counter("order_submit", "failed", venue) == 0
    assert _counter("fill", "recorded", venue) == 0


def test_loopback_submit_ack_fill_and_client_replay_emit_once(
    tmp_path: Path, telemetry: list[dict[str, Any]]
) -> None:
    symbol = "FAKELOOPBACK"
    client = "fake-loopback-client-canary"
    with _open(tmp_path) as desk:
        desk.arm("loopback", "", TS_NS)
        report = desk.submit(symbol, "BUY", 3, 97.5, TS_NS + 1, client_order_id=client)
        assert report["status"] == "filled"
        order = report["order"]
        assert isinstance(order, dict)
        order_id = str(order["order_id"])

        logs_before_replay = list(telemetry)
        metrics_before_replay = render_prometheus()
        replay = desk.submit(symbol, "BUY", 3, 97.5, TS_NS + 2, client_order_id=client)
        assert replay["idempotent"] is True
        assert telemetry == logs_before_replay
        assert render_prometheus() == metrics_before_replay

    assert len(telemetry) == 5
    assert _log_count(telemetry, "order_submit", "attempted", "loopback") == 1
    assert _log_count(telemetry, "order_result", "working", "loopback") == 1
    assert _log_count(telemetry, "order_result", "acknowledged", "loopback") == 1
    assert _log_count(telemetry, "order_result", "filled", "loopback") == 1
    assert _log_count(telemetry, "fill", "recorded", "loopback") == 1
    assert _event_count(telemetry, "order_result") == 3
    assert _counter("order_submit", "attempted", "loopback") == 1
    assert _counter("order_result", "working", "loopback") == 1
    assert _counter("order_result", "acknowledged", "loopback") == 1
    assert _counter("order_result", "filled", "loopback") == 1
    assert _counter("fill", "recorded", "loopback") == 1
    _assert_clean_logs(telemetry, symbol, client, order_id, str(tmp_path))


def test_pretrade_denial_primary_reason_and_replay_suppression(
    tmp_path: Path, telemetry: list[dict[str, Any]]
) -> None:
    rate_root = tmp_path / "rate"
    rate_symbol = "FAKERATEONE"
    denied_rate_symbol = "FAKERATETWO"
    rate_client = "fake-rate-client-canary"
    with _open(rate_root, max_orders_per_window=1) as desk:
        first = desk.submit(
            rate_symbol,
            "BUY",
            1,
            100.0,
            TS_NS,
            client_order_id="fake-rate-first-canary",
        )
        assert first["accepted"] is True
        denied = desk.submit(
            denied_rate_symbol,
            "BUY",
            1,
            100.0,
            TS_NS + 1,
            client_order_id=rate_client,
        )
        assert denied["accepted"] is False
        order = denied["order"]
        assert isinstance(order, dict)
        assert "order_rate" in order["reasons"]

        logs_before_replay = list(telemetry)
        metrics_before_replay = render_prometheus()
        replay = desk.submit(
            denied_rate_symbol,
            "BUY",
            1,
            100.0,
            TS_NS + 2,
            client_order_id=rate_client,
        )
        assert replay["idempotent"] is True
        assert telemetry == logs_before_replay
        assert render_prometheus() == metrics_before_replay

    assert len(telemetry) == 1
    assert _log_count(telemetry, "pretrade_denial", "order_rate", "shadow") == 1
    assert _counter("pretrade_denial", "order_rate", "shadow") == 1
    _assert_no_submit_or_fill(telemetry, "shadow")
    _assert_clean_logs(
        telemetry,
        rate_symbol,
        denied_rate_symbol,
        rate_client,
        "fake-rate-first-canary",
        str(rate_root),
    )

    duplicate_root = tmp_path / "duplicate"
    duplicate_symbol = "FAKEDUPLICATE"
    duplicate_client = "fake-duplicate-client-canary"
    with _open(duplicate_root) as desk:
        first = desk.submit(
            duplicate_symbol,
            "BUY",
            2,
            101.0,
            TS_NS,
            client_order_id="fake-duplicate-first-canary",
        )
        assert first["accepted"] is True
        denied = desk.submit(
            duplicate_symbol,
            "BUY",
            2,
            101.0,
            TS_NS + 1,
            client_order_id=duplicate_client,
        )
        assert denied["accepted"] is False
        order = denied["order"]
        assert isinstance(order, dict)
        assert "duplicate_order" in order["reasons"]

        logs_before_replay = list(telemetry)
        metrics_before_replay = render_prometheus()
        replay = desk.submit(
            duplicate_symbol,
            "BUY",
            2,
            101.0,
            TS_NS + 2,
            client_order_id=duplicate_client,
        )
        assert replay["idempotent"] is True
        assert telemetry == logs_before_replay
        assert render_prometheus() == metrics_before_replay

    assert len(telemetry) == 2
    assert _log_count(telemetry, "pretrade_denial", "order_rate", "shadow") == 1
    assert _log_count(telemetry, "pretrade_denial", "duplicate_order", "shadow") == 1
    assert _counter("pretrade_denial", "order_rate", "shadow") == 1
    assert _counter("pretrade_denial", "duplicate_order", "shadow") == 1
    _assert_no_submit_or_fill(telemetry, "shadow")
    _assert_clean_logs(
        telemetry,
        duplicate_symbol,
        duplicate_client,
        "fake-duplicate-first-canary",
        str(duplicate_root),
    )


def test_kill_reset_only_effective_edges_emit_telemetry(
    tmp_path: Path, telemetry: list[dict[str, Any]]
) -> None:
    actor = "FAKE_OPERATOR_ACTOR_CANARY"
    reason = "FAKE_OPERATOR_REASON_CANARY"
    states: list[object] = []
    with _open(tmp_path) as desk:
        states.append(desk.reset(actor, reason, TS_NS)["kill_latched"])
        states.append(desk.kill(actor, reason, TS_NS + 1)["kill_latched"])
        states.append(desk.kill(actor, reason, TS_NS + 2)["kill_latched"])

    with _open(tmp_path) as desk:
        states.append(desk.status()["kill_latched"])
        states.append(desk.kill(actor, reason, TS_NS + 3)["kill_latched"])
        states.append(desk.reset(actor, reason, TS_NS + 4)["kill_latched"])
        states.append(desk.reset(actor, reason, TS_NS + 5)["kill_latched"])

    assert states == [False, True, True, True, True, False, False]
    assert [(record["event"], record["outcome"], record["venue"]) for record in telemetry] == [
        ("live_desk_kill_switch", "latched", "shadow"),
        ("live_desk_kill_switch", "reset", "shadow"),
    ]
    assert _counter("kill_switch", "latched", "shadow") == 1
    assert _counter("kill_switch", "reset", "shadow") == 1
    _assert_clean_logs(telemetry, actor, reason, str(tmp_path))


def test_drop_fill_and_reject_do_not_repeat_for_persistent_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, telemetry: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    fill_symbol = "FAKEDROPFILL"
    reject_symbol = "FAKEDROPREJECT"
    fill_client = "fake-drop-fill-client-canary"
    reject_client = "fake-drop-reject-client-canary"
    fill_event = "fake-drop-fill-event-canary"
    reject_event = "fake-drop-reject-event-canary"
    reject_reason = "FAKE_BROKER_REJECT_TEXT_CANARY"
    with _open(tmp_path) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        fill_report = desk.submit(
            fill_symbol, "BUY", 2, 41.25, TS_NS + 1, client_order_id=fill_client
        )
        reject_report = desk.submit(
            reject_symbol, "BUY", 3, 43.5, TS_NS + 2, client_order_id=reject_client
        )
        fill_order = fill_report["order"]
        reject_order = reject_report["order"]
        assert isinstance(fill_order, dict)
        assert isinstance(reject_order, dict)
        fill_order_id = str(fill_order["order_id"])
        reject_order_id = str(reject_order["order_id"])

        inbound = tmp_path / "desk" / "inbound"
        inbound.mkdir()
        (inbound / "fill.json").write_text(
            json.dumps(
                {
                    "schema": "live_broker_event.v1",
                    "kind": "fill",
                    "order_id": fill_order_id,
                    "event_id": fill_event,
                    "qty": 2,
                    "price": 41.25,
                }
            ),
            encoding="utf-8",
        )
        (inbound / "reject.json").write_text(
            json.dumps(
                {
                    "schema": "live_broker_event.v1",
                    "kind": "reject",
                    "order_id": reject_order_id,
                    "event_id": reject_event,
                    "reason": reject_reason,
                }
            ),
            encoding="utf-8",
        )
        desk.poll(TS_NS + 3)
        logs_after_first_poll = list(telemetry)
        metrics_after_first_poll = render_prometheus()
        desk.poll(TS_NS + 4)
        assert telemetry == logs_after_first_poll
        assert render_prometheus() == metrics_after_first_poll

    assert len(telemetry) == 7
    assert _log_count(telemetry, "order_submit", "attempted", "drop") == 2
    assert _log_count(telemetry, "order_result", "working", "drop") == 2
    assert _log_count(telemetry, "order_result", "filled", "drop") == 1
    assert _log_count(telemetry, "order_result", "rejected", "drop") == 1
    assert _log_count(telemetry, "fill", "recorded", "drop") == 1
    assert _counter("order_submit", "attempted", "drop") == 2
    assert _counter("order_result", "working", "drop") == 2
    assert _counter("order_result", "filled", "drop") == 1
    assert _counter("order_result", "rejected", "drop") == 1
    assert _counter("fill", "recorded", "drop") == 1
    _assert_clean_logs(
        telemetry,
        fill_symbol,
        reject_symbol,
        fill_client,
        reject_client,
        fill_order_id,
        reject_order_id,
        fill_event,
        reject_event,
        reject_reason,
        str(tmp_path),
    )


def test_orphan_fill_emits_orphan_fault_once_without_accepted_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, telemetry: list[dict[str, Any]]
) -> None:
    monkeypatch.delenv("DIP_LIVE_ARM", raising=False)
    missing_order = "FAKE_MISSING_ORDER_CANARY"
    orphan_event = "FAKE_ORPHAN_EVENT_CANARY"
    with _open(tmp_path) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        inbound = tmp_path / "desk" / "inbound"
        inbound.mkdir()
        (inbound / "orphan.json").write_text(
            json.dumps(
                {
                    "schema": "live_broker_event.v1",
                    "kind": "fill",
                    "order_id": missing_order,
                    "event_id": orphan_event,
                    "qty": 1,
                    "price": 10,
                }
            ),
            encoding="utf-8",
        )
        first = desk.poll(TS_NS + 1)
        assert first["orphan_count"] == 1
        assert first["positions"] == {}
        assert first["orders"] == []
        assert first["cash"] == 1_000_000.0

        logs_after_first_poll = list(telemetry)
        metrics_after_first_poll = render_prometheus()
        second = desk.poll(TS_NS + 2)
        assert second["positions"] == {}
        assert second["cash"] == 1_000_000.0
        assert telemetry == logs_after_first_poll
        assert render_prometheus() == metrics_after_first_poll

    assert len(telemetry) == 2
    assert _log_count(telemetry, "fill", "orphan", "drop") == 1
    assert _log_count(telemetry, "bridge_fault", "orphan", "drop") == 1
    assert _counter("fill", "orphan", "drop") == 1
    assert _counter("bridge_fault", "orphan", "drop") == 1
    assert _counter("fill", "recorded", "drop") == 0
    assert _event_count(telemetry, "order_result") == 0
    assert _event_count(telemetry, "kill_switch") == 0
    assert _counter("kill_switch", "latched", "drop") == 0
    _assert_clean_logs(telemetry, missing_order, orphan_event, str(tmp_path))


def test_disabled_telemetry_preserves_normal_desk_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    reset_metrics()
    sink: list[dict[str, Any]] = []
    set_log_sink(sink)
    try:
        with _open(tmp_path) as desk:
            accepted = desk.submit(
                "FAKEDISABLED",
                "BUY",
                1,
                100.0,
                TS_NS,
                client_order_id="fake-disabled-first-canary",
            )
            assert accepted["accepted"] is True
            assert accepted["status"] == "shadowed"
            denied = desk.submit(
                "FAKEDISABLED",
                "BUY",
                1,
                100.0,
                TS_NS + 1,
                client_order_id="fake-disabled-denial-canary",
            )
            assert denied["accepted"] is False
            order = denied["order"]
            assert isinstance(order, dict)
            assert "duplicate_order" in order["reasons"]
            killed = desk.kill(
                "FAKE_DISABLED_ACTOR_CANARY", "FAKE_DISABLED_REASON_CANARY", TS_NS + 2
            )
            assert killed["kill_latched"] is True
            reset = desk.reset(
                "FAKE_DISABLED_ACTOR_CANARY", "FAKE_DISABLED_REASON_CANARY", TS_NS + 3
            )
            assert reset["kill_latched"] is False

        assert sink == []
        assert render_prometheus() == ""
    finally:
        set_log_sink(None)
        reset_metrics()


def test_bridge_faults_use_bounded_codes_without_raw_exception_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, telemetry: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    transmit_canary = "FAKE_TRANSMIT_EXCEPTION_CANARY"
    with _open(tmp_path / "transmit") as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        bridge = desk._drop
        assert bridge is not None

        def fail_transmit(_ticket: object) -> None:
            raise BridgeError(transmit_canary)

        monkeypatch.setattr(bridge, "transmit", fail_transmit)
        with pytest.raises(LiveDeskError, match=transmit_canary):
            desk.submit("FAKEBRIDGE", "BUY", 1, 100.0, TS_NS + 1)

    assert _counter("order_submit", "attempted", "drop") == 1
    assert _counter("order_submit", "failed", "drop") == 1
    assert _counter("bridge_fault", "transmit", "drop") == 1
    assert _event_count(telemetry, "order_result") == 0
    assert _event_count(telemetry, "kill_switch") == 0
    _assert_clean_logs(telemetry, transmit_canary, "FAKEBRIDGE", str(tmp_path))

    telemetry.clear()
    reset_metrics()
    poll_canary = "FAKE_POLL_FILENAME_CANARY"
    with _open(tmp_path / "poll") as desk:
        inbound = tmp_path / "poll" / "desk" / "inbound"
        inbound.mkdir(parents=True)
        (inbound / f"{poll_canary}.json").write_text("not-json", encoding="utf-8")
        report = desk.poll(TS_NS + 1)
        assert report["accepted"] is False

    assert _counter("bridge_fault", "poll", "drop") == 1
    assert _event_count(telemetry, "kill_switch") == 0
    _assert_clean_logs(telemetry, poll_canary, str(tmp_path))


def test_telemetry_failures_do_not_change_risk_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, telemetry: list[dict[str, Any]]
) -> None:
    calls: list[str] = []

    def fail_metric(**_fields: str) -> None:
        calls.append("metric")
        raise RuntimeError("fake metric failure")

    def fail_log(_event: str, **_fields: object) -> None:
        calls.append("log")
        raise RuntimeError("fake log failure")

    monkeypatch.setattr("quant_fund.live.desk.record_live_desk_event", fail_metric)
    monkeypatch.setattr("quant_fund.live.desk.log_event", fail_log)
    with _open(tmp_path) as desk:
        assert desk.kill("operator", "manual halt", TS_NS)["kill_latched"] is True
        assert desk.reset("operator", "manual reset", TS_NS + 1)["kill_latched"] is False

    assert calls == ["metric", "log", "metric", "log"]
    assert telemetry == []
    assert render_prometheus() == ""
