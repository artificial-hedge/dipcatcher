"""Bounded operational counters for the operator Live Desk."""

from __future__ import annotations

import pytest

from quant_fund.observe.metrics import record_live_desk_event, render_prometheus, reset_metrics


def test_live_desk_counter_is_bounded_and_resettable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    reset_metrics()
    try:
        record_live_desk_event(event="order_submit", outcome="attempted", venue="drop")
        record_live_desk_event(event="order_submit", outcome="attempted", venue="drop")
        record_live_desk_event(event="kill_switch", outcome="latched", venue="shadow")
        expected = (
            "# HELP dipcatcher_live_desk_events_total Operational Live Desk events. "
            "Loopback records are SYNTHETIC; not a live P&L claim.\n"
            "# TYPE dipcatcher_live_desk_events_total counter\n"
            'dipcatcher_live_desk_events_total{event="kill_switch",outcome="latched",'
            'venue="shadow"} 1\n'
            'dipcatcher_live_desk_events_total{event="order_submit",outcome="attempted",'
            'venue="drop"} 2\n'
        )
        assert render_prometheus() == expected
        for event, outcome, venue in (
            ("unbounded-" + "x" * 128, "attempted", "drop"),
            ("order_submit", "unbounded-" + "x" * 128, "drop"),
            ("order_submit", "attempted", "unbounded-" + "x" * 128),
            ("fill", "attempted", "drop"),
        ):
            with pytest.raises(ValueError, match="cannot record live desk event"):
                record_live_desk_event(event=event, outcome=outcome, venue=venue)
        assert render_prometheus() == expected
        reset_metrics()
        assert render_prometheus() == ""
    finally:
        reset_metrics()


def test_live_desk_counter_validates_nothing_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    reset_metrics()
    record_live_desk_event(event="unknown", outcome="unknown", venue="unknown")
    assert render_prometheus() == ""
