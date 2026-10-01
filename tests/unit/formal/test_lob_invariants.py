"""Tests for lob_invariants: executable matching-engine spec."""

from __future__ import annotations

from quant_fund.formal.lob_invariants import (
    LobEvent,
    check_book_trace,
    lobster_events,
    sim_events,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator


def _ev(t: float, kind: str, oid: int, px: int, sz: int, side: int) -> tuple[LobEvent, None]:
    return (LobEvent(t=t, kind=kind, order_id=oid, price=px, size=sz, side=side), None)


def test_clean_trace() -> None:
    evs = [
        _ev(0.0, "submit", 1, 4000, 10, 1),
        _ev(0.1, "submit", 2, 4100, 5, -1),
        _ev(0.2, "exec", 1, 4000, 10, 1),
        _ev(0.3, "delete", 2, 4100, 5, -1),
    ]
    out = check_book_trace(evs)
    assert out["n_violations"] == 0
    assert out["n_events"] == 4


def test_each_invariant_fires() -> None:
    # I2: crossed book after submit
    out = check_book_trace([_ev(0.0, "submit", 1, 4100, 5, -1), _ev(0.1, "submit", 2, 4100, 5, 1)])
    assert out["violation_counts"]["I8_submit_crossed"] == 1
    assert out["violation_counts"]["I2_crossed_book"] == 1
    # I4: exec unknown id
    out = check_book_trace([_ev(0.0, "exec", 99, 4000, 5, 1)])
    assert out["violation_counts"]["I4_exec_unknown_id"] == 1
    # I3: oversize exec
    out = check_book_trace([_ev(0.0, "submit", 1, 4000, 5, 1), _ev(0.1, "exec", 1, 4000, 9, 1)])
    assert out["violation_counts"]["I3_size_overdraw"] == 1
    # I5: price mismatch
    out = check_book_trace([_ev(0.0, "submit", 1, 4000, 5, 1), _ev(0.1, "exec", 1, 4100, 5, 1)])
    assert out["violation_counts"]["I5_exec_price_mismatch"] == 1
    # I6: time goes backwards
    out = check_book_trace([_ev(1.0, "submit", 1, 4000, 5, 1), _ev(0.5, "submit", 2, 4100, 5, -1)])
    assert out["violation_counts"]["I6_time_monotone"] == 1
    # I7: exec behind best (bid 4000 executes while 4050 bid rests)
    out = check_book_trace(
        [
            _ev(0.0, "submit", 1, 4000, 5, 1),
            _ev(0.05, "submit", 9, 4050, 5, 1),
            _ev(0.1, "exec", 1, 4000, 5, 1),
        ]
    )
    assert out["violation_counts"]["I7_exec_behind_best"] == 1


def test_partial_cancel_then_delete() -> None:
    evs = [
        _ev(0.0, "submit", 1, 4000, 10, 1),
        _ev(0.1, "cancel", 1, 4000, 4, 1),
        _ev(0.2, "delete", 1, 4000, 0, 1),  # delete removes remaining 6
        _ev(0.3, "delete", 1, 4000, 0, 1),  # unknown id now
    ]
    out = check_book_trace(evs)
    assert out["violation_counts"]["I4_delete_unknown_id"] == 1
    assert out["n_open_orders"] == 0


def test_lobster_events_kind_map(tmp_path) -> None:
    p = tmp_path / "m.csv"
    p.write_text("34200.0,1,1,10,4000,1\n34200.1,3,1,0,0,1\n")
    evs = list(lobster_events(p))
    assert [e.kind for e, _ in evs] == ["submit", "delete"]


def test_sim_stream_audit() -> None:
    cfg = ZILobConfig(seed=5)
    sim = ZILobSimulator(cfg)
    out = check_book_trace(sim_events(sim, cfg.tick, 4000))
    assert out["n_events"] > 500
    # the sim's own engine should not break structural invariants
    for k in out["violation_counts"]:
        assert not k.startswith(("I1", "I3", "I4", "I5")), out["examples"].get(k)
