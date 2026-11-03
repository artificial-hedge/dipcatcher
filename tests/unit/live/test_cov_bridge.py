"""Fail-closed branches of the desk bridges.

A bridge hands a ticket to a venue (or a file drop) and parses the reports
that come back. Every rejection here is an order that must not be treated as
working, so the errors must name the field and never invent an ack.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.live.bridge import (
    BridgeError,
    DropBridge,
    LoopbackBridge,
    parse_event,
)


def event(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema": "live_broker_event.v1",
        "kind": "fill",
        "order_id": "c1",
        "event_id": "e1",
        "qty": 10.0,
        "price": 100.0,
    }
    base.update(overrides)
    return base


def ticket(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ticket": "order",
        "order_id": "c1",
        "qty": 10.0,
        "limit_price": 100.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------- parse_event


def test_parse_event_accepts_a_well_formed_fill() -> None:
    parsed = parse_event(event(fee=0.5, reason="partial"), synthetic=True, source="loopback")
    assert parsed.kind == "fill"
    assert parsed.qty == 10.0
    assert parsed.fee == 0.5
    assert parsed.synthetic is True


@pytest.mark.parametrize("field", ["schema", "kind", "order_id", "event_id"])
def test_parse_event_requires_its_identity_fields(field: str) -> None:
    with pytest.raises(BridgeError, match=field):
        parse_event(event(**{field: ""}), synthetic=False, source="f.json")


def test_parse_event_rejects_an_unknown_kind() -> None:
    with pytest.raises(BridgeError, match="kind must be"):
        parse_event(event(kind="amend"), synthetic=False, source="f.json")


@pytest.mark.parametrize("value", ["10", True, float("nan"), float("inf")])
def test_parse_event_rejects_non_numeric_and_non_finite_quantities(value: object) -> None:
    with pytest.raises(BridgeError, match="qty"):
        parse_event(event(qty=value), synthetic=False, source="f.json")


def test_parse_event_rejects_a_negative_fee() -> None:
    with pytest.raises(BridgeError, match="fee cannot be negative"):
        parse_event(event(fee=-1.0), synthetic=False, source="f.json")


@pytest.mark.parametrize("overrides", [{"qty": 0.0}, {"price": 0.0}, {"qty": -1.0}])
def test_parse_event_requires_a_positive_fill(overrides: dict[str, object]) -> None:
    with pytest.raises(BridgeError, match="fill needs a positive qty and price"):
        parse_event(event(**overrides), synthetic=False, source="f.json")


def test_parse_event_rejects_a_non_string_reason() -> None:
    with pytest.raises(BridgeError, match="reason must be a string"):
        parse_event(event(reason=7), synthetic=False, source="f.json")


def test_parse_event_allows_a_zero_sized_ack() -> None:
    parsed = parse_event(event(kind="ack", qty=0.0, price=0.0), synthetic=False, source="f.json")
    assert parsed.kind == "ack"
    assert parsed.qty == 0.0


# ------------------------------------------------------------- LoopbackBridge


def test_loopback_transmit_queues_an_ack_and_a_fill() -> None:
    bridge = LoopbackBridge()
    bridge.transmit(ticket())
    queued = bridge.poll()
    assert [item.kind for item in queued] == ["ack", "fill"]
    assert all(item.synthetic for item in queued)
    # poll drains: a second poll must not replay the same fills.
    assert bridge.poll() == []


def test_loopback_cancel_queues_only_a_cancel() -> None:
    bridge = LoopbackBridge()
    bridge.transmit({"ticket": "cancel", "order_id": "c1"})
    queued = bridge.poll()
    assert [item.kind for item in queued] == ["cancel"]
    assert queued[0].event_id == "c-c1"


def test_loopback_refuses_a_ticket_without_an_order_id() -> None:
    bridge = LoopbackBridge()
    with pytest.raises(BridgeError, match="missing order_id"):
        bridge.transmit({"ticket": "order", "qty": 1.0, "limit_price": 1.0})
    with pytest.raises(BridgeError, match="missing order_id"):
        bridge.transmit({"ticket": "cancel", "order_id": ""})
    assert bridge.poll() == []


def test_loopback_refuses_an_unknown_ticket_kind() -> None:
    with pytest.raises(BridgeError, match="order or a cancel"):
        LoopbackBridge().transmit({"ticket": "amend", "order_id": "c1"})


@pytest.mark.parametrize(
    "overrides",
    [{"qty": "10"}, {"limit_price": None}, {"qty": float("nan")}],
)
def test_loopback_refuses_unusable_order_fields(overrides: dict[str, object]) -> None:
    bridge = LoopbackBridge()
    with pytest.raises(BridgeError):
        bridge.transmit(ticket(**overrides))
    assert bridge.poll() == []


# ----------------------------------------------------------------- DropBridge


def test_drop_writes_a_private_non_claiming_ticket(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    bridge.transmit(ticket())
    written = json.loads((tmp_path / "outbound" / "c1.json").read_text(encoding="utf-8"))
    assert written["order_id"] == "c1"
    assert written["live_pnl_claim"] is False
    assert written["would_promote_live"] is False
    assert written["evidence_conditions"] == "blocked"


def test_drop_names_cancel_tickets_and_refuses_the_rest(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    bridge.transmit({"ticket": "cancel", "order_id": "c1"})
    assert (tmp_path / "outbound" / "cancel-c1.json").exists()
    with pytest.raises(BridgeError, match="missing order_id"):
        bridge.transmit({"ticket": "order"})
    with pytest.raises(BridgeError, match="order or a cancel"):
        bridge.transmit({"ticket": "amend", "order_id": "c2"})


def test_drop_poll_is_empty_before_the_operator_writes_anything(tmp_path: Path) -> None:
    assert DropBridge(tmp_path).poll() == []


def test_drop_poll_parses_operator_events(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    inbound = tmp_path / "inbound"
    inbound.mkdir()
    (inbound / "a.json").write_text(json.dumps(event()), encoding="utf-8")
    parsed = bridge.poll()
    assert len(parsed) == 1
    assert parsed[0].order_id == "c1"
    assert parsed[0].synthetic is False


def test_drop_poll_refuses_invalid_json(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    inbound = tmp_path / "inbound"
    inbound.mkdir()
    (inbound / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(BridgeError, match="bad.json: invalid JSON"):
        bridge.poll()


def test_drop_poll_refuses_a_non_object_event(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    inbound = tmp_path / "inbound"
    inbound.mkdir()
    (inbound / "list.json").write_text(json.dumps([event()]), encoding="utf-8")
    with pytest.raises(BridgeError, match="must be a JSON object"):
        bridge.poll()


def test_drop_poll_surfaces_the_field_error_with_the_filename(tmp_path: Path) -> None:
    bridge = DropBridge(tmp_path)
    inbound = tmp_path / "inbound"
    inbound.mkdir()
    (inbound / "evt.json").write_text(json.dumps(event(kind="")), encoding="utf-8")
    with pytest.raises(BridgeError, match="evt.json: kind must be"):
        bridge.poll()
