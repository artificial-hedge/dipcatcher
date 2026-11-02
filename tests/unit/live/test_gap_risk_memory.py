"""Unit tests for ``quant_fund.live.risk_memory``.

``risk_memory`` is the bridge that lets the pre-trade rate, duplicate,
session-start and peak-memory limits survive a CLI process boundary. The bug
it exists to fix was that a fresh process built an *empty* engine, so those
limits silently reset to zero every time the desk was restarted.

The contract pinned here:

* a snapshot exports exactly the counters a later process must enforce;
* restoring that snapshot into a fresh engine reproduces the same state;
* malformed memory **fails closed** — it never restores a partial state and
  never silently accepts a truncated ring.
"""

from __future__ import annotations

import copy

import pytest

from quant_fund.live.errors import LiveGateError
from quant_fund.live.risk_memory import export_pretrade, restore_pretrade
from tests.unit.pretrade.support import TS_NS, arm, make_engine, order


def _armed_engine(symbol: str = "A"):
    engine = make_engine()
    sid = arm(engine, symbol)
    return engine, sid


def _busy_engine():
    """An engine with a filled order, a message and a duplicate fingerprint."""
    engine, sid = _armed_engine()
    engine.check(order(sid, qty=10.0, px=100.0, ts_ns=TS_NS))
    engine.check(order(sid, qty=10.0, px=100.0, ts_ns=TS_NS + 1))
    engine.check(order(sid, qty=10.0, px=100.0, ts_ns=TS_NS + 2))
    return engine


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_export_carries_the_counters_a_later_process_needs() -> None:
    engine, sid = _armed_engine()
    blob = export_pretrade(engine)
    assert blob["symbols"] == ["A"]
    assert isinstance(blob["session_start"], float)
    assert isinstance(blob["peak"], float)
    assert isinstance(blob["last_ts"], int)
    assert isinstance(blob["order_ts"], list)
    assert isinstance(blob["message_ts"], list)
    assert isinstance(blob["dupes"], list)


def test_export_of_an_idle_engine_is_empty_but_well_formed() -> None:
    blob = export_pretrade(make_engine())
    assert blob["symbols"] == []
    assert blob["order_ts"] == []
    assert blob["message_ts"] == []
    assert blob["dupes"] == []


def test_export_records_the_order_ring_the_engine_actually_holds() -> None:
    engine = _busy_engine()
    blob = export_pretrade(engine)
    assert blob["order_ts"], "a processed order must be visible in the export"
    assert all(isinstance(stamp, int) for stamp in blob["order_ts"])


# --------------------------------------------------------------------------- #
# Restore
# --------------------------------------------------------------------------- #


def test_restore_reproduces_the_same_state_in_a_fresh_engine() -> None:
    """This is the whole point: the limits must not reset across processes."""
    source = _busy_engine()
    blob = export_pretrade(source)

    target = make_engine()
    restore_pretrade(target, blob)
    assert export_pretrade(target) == blob


def test_restore_carries_the_symbol_set_over() -> None:
    source = make_engine()
    arm(source, "AAA")
    arm(source, "BBB")
    blob = export_pretrade(source)

    target = make_engine()
    restore_pretrade(target, blob)
    assert target._ids == {"AAA": 0, "BBB": 1}


def test_restore_rejects_a_non_mapping_snapshot() -> None:
    target = make_engine()
    for bad in (None, [], "state", 7):
        with pytest.raises(LiveGateError, match="malformed"):
            restore_pretrade(target, bad)


def test_restore_rejects_a_malformed_symbol_list() -> None:
    target = make_engine()
    blob = export_pretrade(make_engine())
    for bad in (None, "A", {"A": 1}, 3):
        broken = {**blob, "symbols": bad}
        with pytest.raises(LiveGateError, match="symbols are malformed"):
            restore_pretrade(target, broken)


def test_restore_rejects_a_non_string_symbol() -> None:
    target = make_engine()
    blob = {**export_pretrade(make_engine()), "symbols": [123]}
    with pytest.raises(LiveGateError, match="symbol could not be restored"):
        restore_pretrade(target, blob)


def test_restore_rejects_malformed_floors() -> None:
    target = make_engine()
    base = export_pretrade(make_engine())
    for key in ("session_start", "peak", "last_ts"):
        broken = {**base, key: "not-a-number"}
        with pytest.raises(LiveGateError, match="floors are malformed"):
            restore_pretrade(target, broken)


def test_restore_rejects_malformed_ring_entries() -> None:
    target = make_engine()
    base = export_pretrade(make_engine())
    for key, label in (("order_ts", "order timestamps"), ("message_ts", "message timestamps")):
        broken = {**base, key: "nope"}
        with pytest.raises(LiveGateError, match=label):
            restore_pretrade(target, broken)
        broken = {**base, key: ["not-an-int"]}
        with pytest.raises(LiveGateError, match=label):
            restore_pretrade(target, broken)


def test_restore_rejects_malformed_duplicate_entries() -> None:
    target = make_engine()
    base = export_pretrade(make_engine())
    for bad in ("nope", [[1]], [[1, 2, 3]], [["a", "b"]]):
        broken = {**base, "dupes": bad}
        with pytest.raises(LiveGateError, match="duplicates are malformed"):
            restore_pretrade(target, broken)


def test_restore_rejects_a_ring_larger_than_the_capacity() -> None:
    """A snapshot whose ring does not fit is refused rather than truncated."""
    target = make_engine()
    capacity = int(target.book.ring_cap)
    base = export_pretrade(make_engine())
    broken = {**base, "order_ts": list(range(capacity + 1))}
    with pytest.raises(LiveGateError, match="ring does not fit"):
        restore_pretrade(target, broken)


def test_restore_leaves_the_target_untouched_when_it_fails_closed() -> None:
    """A rejected snapshot must not half-apply to the live engine."""
    target, sid = _armed_engine()
    before = export_pretrade(target)
    broken = {**before, "symbols": "not-a-list"}
    with pytest.raises(LiveGateError):
        restore_pretrade(target, broken)
    assert export_pretrade(target)["symbols"] == before["symbols"]


def test_export_is_a_snapshot_not_a_live_view() -> None:
    """Mutating the exported blob must not change the engine's state."""
    engine, sid = _armed_engine()
    blob = export_pretrade(engine)
    mutated = copy.deepcopy(blob)
    mutated["symbols"].append("GHOST")
    mutated["peak"] = 1.0
    assert export_pretrade(engine)["symbols"] == blob["symbols"]
    assert export_pretrade(engine)["peak"] == blob["peak"]
