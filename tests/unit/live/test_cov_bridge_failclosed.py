"""Inbound broker reports: malformed events fail closed, never partially applied.

The drop bridge is the only place an operator's broker process hands results
back to the desk. A bad file must raise instead of yielding a half-parsed
event, and an outbound ticket must carry the blocked honesty flags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quant_fund.live.bridge import (
    BridgeError,
    DropBridge,
    LoopbackBridge,
    parse_event,
)


def _event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema": "live_broker_event.v1",
        "kind": "ack",
        "order_id": "ord-1",
        "event_id": "evt-1",
    }
    event.update(overrides)
    return event


def test_a_valid_ack_parses() -> None:
    event = parse_event(_event(), synthetic=False, source="e.json")
    assert (event.kind, event.order_id, event.event_id) == ("ack", "ord-1", "evt-1")
    assert event.synthetic is False


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"kind": "ack", "order_id": "o", "event_id": "e"}, "schema must be"),
        (_event(schema="live_broker_event.v2"), "schema must be"),
        (_event(kind="hold"), "kind must be ack"),
        (_event(kind=None), "kind must be ack"),
        (_event(kind=1), "kind must be ack"),
        (_event(order_id=""), "order_id is required"),
        (_event(order_id=7), "order_id is required"),
        (_event(event_id=""), "event_id is required"),
        (_event(event_id=None), "event_id is required"),
    ],
)
def test_malformed_envelopes_are_refused(raw: dict[str, Any], match: str) -> None:
    with pytest.raises(BridgeError, match=match):
        parse_event(raw, synthetic=False, source="e.json")


@pytest.mark.parametrize("field", ["qty", "price", "fee"])
@pytest.mark.parametrize("value", ["1.0", True, None, float("nan"), float("inf")])
def test_non_numeric_or_non_finite_numbers_are_refused(field: str, value: object) -> None:
    with pytest.raises(BridgeError, match=f"{field} must be"):
        parse_event(_event(**{field: value}), synthetic=False, source="e.json")


def test_a_negative_fee_is_refused() -> None:
    with pytest.raises(BridgeError, match="fee cannot be negative"):
        parse_event(_event(fee=-0.01), synthetic=False, source="e.json")


@pytest.mark.parametrize("qty, price", [(0.0, 10.0), (10.0, 0.0), (-1.0, 10.0)])
def test_a_fill_needs_a_positive_quantity_and_price(qty: float, price: float) -> None:
    with pytest.raises(BridgeError, match="fill needs a positive qty and price"):
        parse_event(_event(kind="fill", qty=qty, price=price), synthetic=False, source="e.json")


def test_a_non_string_reason_is_refused() -> None:
    with pytest.raises(BridgeError, match="reason must be a string"):
        parse_event(_event(reason=42), synthetic=False, source="e.json")


def test_a_missing_inbound_directory_polls_empty(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    assert bridge.poll() == []


def test_invalid_json_in_the_drop_is_refused(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    bridge.inbound.mkdir(parents=True)
    (bridge.inbound / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(BridgeError, match="bad.json: invalid JSON"):
        bridge.poll()


def test_a_non_object_drop_file_is_refused(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    bridge.inbound.mkdir(parents=True)
    (bridge.inbound / "list.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(BridgeError, match="event must be a JSON object"):
        bridge.poll()


def test_a_valid_drop_file_is_parsed_as_a_real_event(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    bridge.inbound.mkdir(parents=True)
    (bridge.inbound / "a.json").write_text(json.dumps(_event()), encoding="utf-8")
    events = bridge.poll()
    assert [event.event_id for event in events] == ["evt-1"]
    assert events[0].synthetic is False


def test_one_bad_drop_file_does_not_drop_the_parse_for_good(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    bridge.inbound.mkdir(parents=True)
    (bridge.inbound / "a.json").write_text(json.dumps(_event()), encoding="utf-8")
    (bridge.inbound / "b.json").write_text(json.dumps(_event(kind="hold")), encoding="utf-8")
    with pytest.raises(BridgeError, match="b.json: kind must be"):
        bridge.poll()


def test_an_outbound_ticket_carries_the_blocked_honesty_flags(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    bridge.transmit({"ticket": "order", "order_id": "ord-1", "qty": 1.0, "limit_price": 10.0})
    payload = json.loads((bridge.outbound / "ord-1.json").read_text(encoding="utf-8"))
    assert payload["live_pnl_claim"] is False
    assert payload["would_promote_live"] is False
    assert payload["evidence_conditions"] == "blocked"


def test_a_cancel_ticket_is_named_as_a_cancel(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    bridge.transmit({"ticket": "cancel", "order_id": "ord-2"})
    assert (bridge.outbound / "cancel-ord-2.json").exists()


@pytest.mark.parametrize(
    "ticket", [{}, {"ticket": "order"}, {"order_id": "ord-3", "ticket": "hold"}]
)
def test_an_unusable_outbound_ticket_is_refused(tmp_path: Path, ticket: dict[str, Any]) -> None:
    bridge = DropBridge(tmp_path)
    with pytest.raises(BridgeError, match="missing order_id|must be an order or a cancel"):
        bridge.transmit(ticket)
    assert not bridge.outbound.exists()


def test_loopback_queues_a_synthetic_ack_and_fill_then_a_cancel() -> None:
    bridge = LoopbackBridge()
    bridge.transmit({"ticket": "order", "order_id": "ord-1", "qty": 2, "limit_price": 9.5})
    bridge.transmit({"ticket": "cancel", "order_id": "ord-2"})
    events = bridge.poll()
    assert [(event.kind, event.order_id) for event in events] == [
        ("ack", "ord-1"),
        ("fill", "ord-1"),
        ("cancel", "ord-2"),
    ]
    assert all(event.synthetic for event in events)
    assert events[1].qty == 2.0
    assert events[1].price == 9.5
    assert bridge.poll() == []


def test_loopback_refuses_an_unusable_ticket() -> None:
    bridge = LoopbackBridge()
    with pytest.raises(BridgeError, match="missing order_id"):
        bridge.transmit({"ticket": "order"})
    with pytest.raises(BridgeError, match="must be an order or a cancel"):
        bridge.transmit({"ticket": "hold", "order_id": "ord-1"})
    with pytest.raises(BridgeError, match="loopback qty must be a number"):
        bridge.transmit({"ticket": "order", "order_id": "ord-1", "qty": "1", "limit_price": 10})
    assert bridge.poll() == []
