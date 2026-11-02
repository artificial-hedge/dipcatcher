"""Fail-closed checks for the operator paper session.

SYNTHETIC marks and fills are correctness fixtures, never market evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.execution.order_recon import LiveEndpointRefused
from quant_fund.paper.trading_session import (
    PaperTradingSession,
    SessionCommandError,
    SessionLimits,
)

T0 = "2026-10-13T14:30:00+00:00"


def _limits(**overrides: float) -> SessionLimits:
    payload = {
        "max_order_quantity": 100.0,
        "max_order_notional": 100_000.0,
        "max_position_notional": 100_000.0,
        "max_gross_notional": 100_000.0,
        "max_loss_fraction": 0.1,
    }
    payload.update(overrides)
    return SessionLimits(**payload)


def _session(tmp_path: Path, **limits: float) -> PaperTradingSession:
    return PaperTradingSession(
        state_path=tmp_path / "book.json",
        starting_cash=1_000.0,
        limits=_limits(**limits),
    )


def test_loss_limit_blocks_the_next_order_and_cancel_still_works(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.apply_command({"op": "mark", "security_id": "AAPL", "price": 100.0})
    assert (
        session.apply_command(
            {
                "op": "submit",
                "order_id": "o1",
                "security_id": "AAPL",
                "side": "BUY",
                "quantity": 10,
            }
        )["verdict"]
        == "accepted"
    )
    assert (
        session.apply_command(
            {
                "op": "fill",
                "fill_id": "f1",
                "order_id": "o1",
                "security_id": "AAPL",
                "quantity": 5,
                "price": 100.0,
                "fill_time": T0,
            }
        )["verdict"]
        == "accepted"
    )
    session.apply_command({"op": "mark", "security_id": "AAPL", "price": 50.0})
    refused = session.apply_command(
        {
            "op": "submit",
            "order_id": "o2",
            "security_id": "AAPL",
            "side": "BUY",
            "quantity": 1,
        }
    )
    assert refused["verdict"] == "rejected"
    assert refused["reason"] == "max_loss"
    canceled = session.apply_command({"op": "cancel", "order_id": "o1"})
    assert canceled["verdict"] == "cancelled"
    view = session.reconcile_view()
    assert view["computed_positions"]["AAPL"] == pytest.approx(5.0)
    assert view["live_orders_enabled"] is False
    assert view["would_promote_live"] is False


def test_identical_order_id_stays_a_duplicate(tmp_path: Path) -> None:
    session = _session(tmp_path, max_loss_fraction=1.0)
    session.apply_command({"op": "mark", "security_id": "AAPL", "price": 10.0})
    command = {
        "op": "submit",
        "order_id": "o1",
        "security_id": "AAPL",
        "side": "BUY",
        "quantity": 1,
    }
    assert session.apply_command(command)["verdict"] == "accepted"
    assert session.apply_command(command)["verdict"] == "duplicate_order"
    conflict = dict(command)
    conflict["quantity"] = 2
    rejected = session.apply_command(conflict)
    assert rejected["verdict"] == "rejected"
    assert "conflicts" in rejected["reason"]
    assert session.reconcile_view()["n_orders"] == 1


def test_loss_fraction_above_one_is_rejected() -> None:
    with pytest.raises(SessionCommandError, match="at most 1"):
        SessionLimits(max_loss_fraction=1.5)


def test_paper_session_still_refuses_a_live_endpoint(tmp_path: Path) -> None:
    with pytest.raises(LiveEndpointRefused):
        PaperTradingSession(state_path=tmp_path / "book.json", endpoint_kind="alpaca-live")
