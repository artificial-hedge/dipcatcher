"""Live-gateway exposure stays inside the armed ceilings. SYNTHETIC, no sockets.

Hypothesis is derandomized. Splits, rapid repeats, and partial fills are
correctness checks: ``live_pnl_claim`` stays false and nothing here is a
research receipt.
"""

from __future__ import annotations

from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.live.session import LiveSession
from tests.unit.fault.support import TS, FaultBroker, arm, buy, open_fault_session

_FAST = settings(derandomize=True, max_examples=20, deadline=None)
_STEP = 2_100_000_000
_PRICE = 10.0


def _exposure(session: LiveSession) -> tuple[float, float, dict[str, float]]:
    account = session.broker.account()
    reserved, _cash = session._working_exposure()
    gross = 0.0
    net = 0.0
    qty: dict[str, float] = {}
    for symbol in set(account.positions) | set(reserved):
        held = float(account.positions.get(symbol, 0.0)) + float(reserved.get(symbol, 0.0))
        qty[symbol] = held
        mark = float(session.marks.get(symbol) or 0.0)
        notion = held * mark
        gross += abs(notion)
        net += notion
    return gross, net, qty


def _inside(session: LiveSession) -> None:
    limits = session.pretrade.limits
    gross, net, qty = _exposure(session)
    account = session.broker.account()
    nav = account.equity if account.equity > 0.0 else 1.0
    for held in qty.values():
        assert abs(held) <= limits.max_position_quantity + 1e-6
    for symbol, held in qty.items():
        mark = float(session.marks.get(symbol) or 0.0)
        assert abs(held * mark) <= limits.max_position_notional + 1e-4
        assert abs(held * mark) <= limits.max_name_concentration * nav + 1e-2
    assert gross <= limits.max_gross_notional + 1e-2
    assert abs(net) <= limits.max_net_notional + 1e-2
    assert session.status()["live_pnl_claim"] is False


@given(st.lists(st.integers(min_value=1, max_value=8_000), min_size=1, max_size=5))
@_FAST
def test_split_and_rapid_buys_stay_inside_live_ceilings(qtys: list[int]) -> None:
    with TemporaryDirectory() as raw:
        broker = FaultBroker()
        session = open_fault_session(raw, broker)  # type: ignore[arg-type]
        arm(session)
        stamp = TS
        seen = broker.submit_count
        for index, qty in enumerate(qtys):
            result = buy(
                session,
                quantity=float(qty),
                limit_price=_PRICE,
                mark=_PRICE,
                client_order_id=f"slice-{index}-{qty}",
                at_ns=stamp,
            )
            assert result["live_pnl_claim"] is False
            if result["ok"]:
                assert result["broker_called"] is True
                assert qty <= session.pretrade.limits.max_order_quantity
                assert qty * _PRICE <= session.pretrade.limits.max_order_notional
                assert broker.submit_count == seen + 1
                seen = broker.submit_count
            else:
                assert result["broker_called"] is False
                assert broker.submit_count == seen
            _inside(session)
            stamp += _STEP


@given(st.integers(min_value=1, max_value=9_999))
@_FAST
def test_partial_fill_keeps_the_working_remainder_inside_the_position_cap(filled: int) -> None:
    with TemporaryDirectory() as raw:
        broker = FaultBroker()
        session = open_fault_session(raw, broker)  # type: ignore[arg-type]
        arm(session)
        first = buy(
            session,
            quantity=10_000.0,
            limit_price=_PRICE,
            mark=_PRICE,
            client_order_id="partial-open",
            at_ns=TS,
        )
        assert first["ok"] is True
        assert (
            broker.deliver("partial-open", float(filled), _PRICE, TS, "partial-fill") == "applied"
        )
        topped = buy(
            session,
            quantity=10_000.0,
            limit_price=_PRICE,
            mark=_PRICE,
            client_order_id="partial-top",
            at_ns=TS + _STEP,
        )
        assert topped["ok"] is True
        assert topped["broker_called"] is True
        overflow = buy(
            session,
            quantity=1.0,
            limit_price=_PRICE,
            mark=_PRICE,
            client_order_id="partial-over",
            at_ns=TS + 2 * _STEP,
        )
        assert overflow["ok"] is False
        assert overflow["broker_called"] is False
        _inside(session)
        _gross, _net, qty = _exposure(session)
        assert qty.get("AAPL", 0.0) <= 20_000.0 + 1e-6
