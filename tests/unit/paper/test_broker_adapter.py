"""Crash-resume recovery tests for the paper-only broker adapter.

SYNTHETIC fills/prices only (correctness tests, never market evidence). Proves
kill/resume mid-run NAV seam parity and NO duplicate orders, plus the hard
live-endpoint refusal. NAV is a simulated accounting identity, never live P&L.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from math import inf, nan

import pytest

from quant_fund.execution.order_recon import Fill, LiveEndpointRefused, Order, OrderLedgerError
from quant_fund.paper.broker_adapter import PaperBrokerAdapter
from quant_fund.schemas.errors import KillSwitchActive

T0 = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


def _order(oid: str, sec: str = "SEC_A", side: str = "BUY", qty: float = 10.0) -> Order:
    return Order(order_id=oid, security_id=sec, side=side, quantity=qty)  # type: ignore[arg-type]


def _fill(fid: str, oid: str, qty: float, price: float, t: datetime = T0) -> Fill:
    return Fill(
        fill_id=fid, order_id=oid, security_id="SEC_A", quantity=qty, price=price, fill_time=t
    )


def test_live_endpoint_refused_at_construction(tmp_path) -> None:
    with pytest.raises(LiveEndpointRefused):
        PaperBrokerAdapter(state_path=tmp_path / "s.json", endpoint_kind="live")


def test_kill_resume_nav_seam_parity_and_no_duplicate_orders(tmp_path) -> None:
    path = tmp_path / "state.json"
    run1 = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    run1.submit(_order("o1", qty=10.0))
    run1.apply_fill(_fill("f1", "o1", qty=10.0, price=10.0))
    run1.mark("SEC_A", 11.0)
    nav_before = run1.nav
    # "kill" mid-run: trip the switch, checkpoint, then drop the process.
    run1.trip_kill("mid_run_crash")
    run1.checkpoint()

    run2, seam = PaperBrokerAdapter.resume(path)
    assert seam["nav_seam_parity"] is True
    assert seam["ok"] is True
    assert seam["duplicate_orders"] == 0
    assert abs(nav_before - seam["persisted_nav"]) <= 1e-6
    assert abs(run2.nav - nav_before) <= 1e-6
    # the resumed run remains halted after the prior kill trip
    assert run2._ledger.kill_state == "HALT_NEW_ORDERS"
    report = run2.seam_report()
    assert report["live_pnl_claim"] is False


def test_no_duplicate_orders_across_resume(tmp_path) -> None:
    path = tmp_path / "state.json"
    run1 = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    assert run1.submit(_order("o1")) == "accepted"
    assert run1.submit(_order("o1")) == "duplicate_order"
    run1.checkpoint()

    run2, seam = PaperBrokerAdapter.resume(path)
    assert run2._ledger.kill_state == "HALT_NEW_ORDERS"
    with pytest.raises(KillSwitchActive):
        run2.submit(_order("new"))
    run2.clear_kill()
    # re-submitting the same order after resume stays idempotent (no dupes)
    assert run2.submit(_order("o1")) == "duplicate_order"
    assert seam["duplicate_orders"] == 0
    # replaying the same fill is idempotent too
    assert run2.apply_fill(_fill("f1", "o1", qty=10.0, price=10.0)) == "accepted"
    assert run2.apply_fill(_fill("f1", "o1", qty=10.0, price=10.0)) == "duplicate_fill"


def test_resume_continues_and_totals_are_consistent(tmp_path) -> None:
    path = tmp_path / "state.json"
    run1 = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    run1.submit(_order("o1", qty=5.0))
    run1.apply_fill(_fill("f1", "o1", qty=5.0, price=10.0))
    run1.mark("SEC_A", 10.0)
    run1.checkpoint()

    run2, _ = PaperBrokerAdapter.resume(path)
    run2.clear_kill()
    run2.submit(_order("o2", qty=5.0))
    run2.apply_fill(_fill("f2", "o2", qty=5.0, price=10.0, t=T1))
    run2.mark("SEC_A", 12.0)
    rep = run2.checkpoint()
    assert rep["ok"] is True
    assert rep["nav_seam_parity"] is True
    # 1000 cash - 100 spent on 10 shares @10; mark 12 -> nav 1000 - 100 + 120
    assert abs(run2.nav - 1020.0) <= 1e-6


def test_seam_reports_research_only(tmp_path) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path, starting_cash=0.0)
    run.checkpoint()
    report = run.seam_report()
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("endpoint_kind", "simulated"),
        ("kill_state", "UNKNOWN"),
        ("research_only", False),
        ("live_pnl_claim", True),
        ("schema", "other"),
    ],
)
def test_resume_refuses_changed_checkpoint_control_fields(tmp_path, field, value) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path)
    run.trip_kill("halt")
    run.checkpoint()
    state = json.loads(path.read_text(encoding="utf-8"))
    state[field] = value
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(OrderLedgerError):
        PaperBrokerAdapter.resume(path)


def test_resume_refuses_changed_nav_or_nonfinite_state(tmp_path) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    run.checkpoint()
    state = json.loads(path.read_text(encoding="utf-8"))
    state["nav"] = 2000.0
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(OrderLedgerError, match="seam verification"):
        PaperBrokerAdapter.resume(path)
    state["nav"] = nan
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(OrderLedgerError):
        PaperBrokerAdapter.resume(path)


def test_nonfinite_mark_cannot_poison_paper_nav(tmp_path) -> None:
    run = PaperBrokerAdapter(state_path=tmp_path / "state.json", starting_cash=1000.0)
    with pytest.raises(OrderLedgerError):
        run.mark("SEC_A", nan)
    assert run.nav == 1000.0


def test_resume_stays_halted_even_if_checkpoint_claims_kill_was_cleared(tmp_path) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path)
    run.trip_kill("halt")
    run.checkpoint()
    state = json.loads(path.read_text(encoding="utf-8"))
    state["kill_state"] = "ENABLED"
    path.write_text(json.dumps(state), encoding="utf-8")
    resumed, report = PaperBrokerAdapter.resume(path)
    assert report["ok"] is True
    assert resumed._ledger.kill_state == "HALT_NEW_ORDERS"
    with pytest.raises(KillSwitchActive):
        resumed.submit(_order("new"))


def test_orphan_fill_is_preserved_and_blocks_resume(tmp_path) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path)
    assert run.apply_fill(_fill("orphan", "missing", qty=1.0, price=10.0)) == "orphan_fill"
    report = run.checkpoint()
    assert report["ok"] is False
    assert "orphan_fills" in report["errors"]
    state = json.loads(path.read_text(encoding="utf-8"))
    assert [fill["fill_id"] for fill in state["fills"]] == ["orphan"]
    with pytest.raises(OrderLedgerError, match="seam verification"):
        PaperBrokerAdapter.resume(path)


def test_duplicate_fill_in_checkpoint_blocks_resume(tmp_path) -> None:
    """A duplicated persisted fill must not turn into a green seam report."""
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    run.submit(_order("o1"))
    run.apply_fill(_fill("f1", "o1", qty=10.0, price=10.0))
    run.checkpoint()
    state = json.loads(path.read_text(encoding="utf-8"))
    state["fills"].append(dict(state["fills"][0]))
    path.write_text(json.dumps(state), encoding="utf-8")

    report = run.seam_report()
    assert report["ok"] is False
    assert "duplicate_fills" in report["errors"]
    with pytest.raises(OrderLedgerError, match="duplicate_fills"):
        PaperBrokerAdapter.resume(path)


def test_nonfinite_nav_cannot_publish_checkpoint(tmp_path) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path)
    run.mark("SEC_A", 1e308)
    run.submit(_order("o1", qty=1e308))
    run.apply_fill(_fill("f1", "o1", qty=1e308, price=1e-308))
    with pytest.raises(OrderLedgerError, match="NAV must be finite"):
        run.checkpoint()
    assert not path.exists()


def test_mark_that_would_overflow_nav_is_rolled_back(tmp_path) -> None:
    run = PaperBrokerAdapter(state_path=tmp_path / "state.json")
    run.submit(_order("o1", qty=1e308))
    run.apply_fill(_fill("f1", "o1", qty=1e308, price=1e-308))
    before = run.nav
    with pytest.raises(OrderLedgerError, match="NAV must be finite"):
        run.mark("SEC_A", 1e308)
    assert run.nav == before


@pytest.mark.parametrize("nav_tolerance", [-1.0, nan, inf, True, "1e-6", 10**400])
def test_invalid_nav_tolerance_refused_at_construction(tmp_path, nav_tolerance) -> None:
    with pytest.raises(OrderLedgerError, match="nav_tolerance"):
        PaperBrokerAdapter(state_path=tmp_path / "state.json", nav_tolerance=nav_tolerance)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("orders", "order_id", 123),
        ("orders", "quantity", "10"),
        ("orders", "expects_fill", "false"),
        ("orders", "submitted_time", False),
        ("fills", "fill_id", 123),
        ("fills", "quantity", "10"),
        ("fills", "price", "10"),
        ("fills", "seq", "1"),
        ("fills", "fill_time", 123),
    ],
)
def test_checkpoint_does_not_coerce_order_or_fill_fields(tmp_path, section, field, value) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    run.submit(_order("o1"))
    run.apply_fill(_fill("f1", "o1", qty=10.0, price=10.0))
    run.checkpoint()
    state = json.loads(path.read_text(encoding="utf-8"))
    state[section][0][field] = value
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(OrderLedgerError):
        PaperBrokerAdapter.resume(path)


@pytest.mark.parametrize("price", ["11", -1.0, True])
def test_checkpoint_does_not_coerce_mark_prices(tmp_path, price) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path)
    run.mark("SEC_A", 11.0)
    run.checkpoint()
    state = json.loads(path.read_text(encoding="utf-8"))
    state["marks"]["SEC_A"] = price
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(OrderLedgerError):
        PaperBrokerAdapter.resume(path)
