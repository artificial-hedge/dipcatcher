"""Live quote and corporate-action invariants. SYNTHETIC, no sockets.

Hypothesis is derandomized. Nothing here is a research receipt.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.live.brokers import LiveOrder
from quant_fund.live.errors import LiveGateError
from quant_fund.live.market_data import mapping_is_bijective
from quant_fund.live.session import LiveSession
from tests.unit.fault.support import TS, FaultBroker, arm, buy, open_fault_session

_FAST = settings(derandomize=True, max_examples=20, deadline=None)
_STEP = 2_100_000_000


class _Fills(FaultBroker):
    """Paper venue that prints the whole order on submit. Still no socket."""

    def __init__(self) -> None:
        super().__init__()
        self.fill_on_submit = True

    def submit(self, order: LiveOrder):  # type: ignore[no-untyped-def]
        ack = super().submit(order)
        if not self.fill_on_submit or ack.status != "accepted":
            return ack
        self.deliver(
            order.client_order_id,
            order.quantity,
            order.limit_price,
            order.fill_time_ns,
            f"fill-{order.client_order_id}",
        )
        return self._acks[order.client_order_id]


def _session(root: Path, broker: FaultBroker | None = None) -> LiveSession:
    venue = broker if broker is not None else FaultBroker()
    session = open_fault_session(root, venue)
    arm(session)
    return session


@given(st.sampled_from([0.0, -1.0, float("nan"), float("inf"), float("-inf"), 200.0]))
@_FAST
def test_bad_prices_never_replace_the_mark_or_authorize(price: float) -> None:
    with TemporaryDirectory() as raw:
        broker = FaultBroker()
        session = _session(Path(raw), broker)
        session.mark(symbol="AAPL", price=100.0, actor="ada", at_ns=TS)
        with pytest.raises(LiveGateError):
            session.mark(symbol="AAPL", price=price, actor="ada", at_ns=TS + 1)
        assert session.marks["AAPL"] == 100.0
        assert session.kill_state["latched"] is False
        try:
            denied = buy(
                session,
                mark=price,
                limit_price=100.0,
                client_order_id="bad-mark",
                at_ns=TS + 2,
            )
        except LiveGateError:
            assert session.marks["AAPL"] == 100.0
            assert broker.submit_count == 0
            return
        assert denied["broker_called"] is False
        assert session.marks["AAPL"] == 100.0
        assert broker.submit_count == 0


@given(st.lists(st.integers(min_value=0, max_value=6), min_size=1, max_size=4))
@_FAST
def test_sequence_faults_trip_stale_without_filling_the_mark(seqs: list[int]) -> None:
    with TemporaryDirectory() as raw:
        broker = FaultBroker()
        session = _session(Path(raw), broker)
        session.mark(symbol="AAPL", price=100.0, actor="ada", at_ns=TS)
        first = session.ingest_quote(
            symbol="AAPL", bid=100.0, ask=101.0, seq=1, actor="ada", at_ns=TS + 1
        )
        assert first["ok"] is True
        anchor = float(session.marks["AAPL"])
        stamp = TS + 2
        for index, seq in enumerate(seqs):
            quote = session.ingest_quote(
                symbol="AAPL",
                bid=100.0,
                ask=101.0,
                seq=seq,
                actor="ada",
                at_ns=stamp,
            )
            stamp += 1
            if seq != index + 2:
                assert quote["ok"] is False
                assert quote["stale"] is True
                assert session.marks["AAPL"] == anchor
                denied = buy(
                    session,
                    client_order_id=f"gap-{index}",
                    at_ns=stamp,
                )
                stamp += 1
                assert denied["broker_called"] is False
                assert "stale_data" in denied["reasons"]
                assert session.marks["AAPL"] == anchor
                assert broker.submit_count == 0
                break


@given(st.floats(min_value=1.01, max_value=4.0, allow_nan=False, allow_infinity=False))
@_FAST
def test_splits_keep_equity_and_do_not_latch(ratio: float) -> None:
    with TemporaryDirectory() as raw:
        broker = _Fills()
        session = _session(Path(raw), broker)
        assert buy(session, quantity=100.0, client_order_id="lot")["ok"] is True
        before = broker.account().equity
        peak = float(session.engine.book.peak)
        plan = session.apply_corporate_action(
            kind="split",
            symbol="AAPL",
            actor="ada",
            ratio=ratio,
            at_ns=TS + _STEP,
        )
        assert plan["live_pnl_claim"] is False
        assert broker.account().equity == pytest.approx(before)
        assert session.kill_state["latched"] is False
        assert float(session.engine.book.peak) == pytest.approx(peak)
        if plan["ok"]:
            assert broker.positions["AAPL"] == pytest.approx(100.0 * ratio)
            assert session.marks["AAPL"] == pytest.approx(100.0 / ratio)
        else:
            assert plan["reason"] == "position_limit"
            assert broker.positions["AAPL"] == pytest.approx(100.0)
            assert session.marks["AAPL"] == 100.0


@given(
    st.lists(
        st.tuples(
            st.sampled_from(["AAPL", "MSFT", "IBM"]),
            st.sampled_from(["AAPL", "MSFT", "IBM", "APL"]),
        ),
        min_size=1,
        max_size=5,
    )
)
@_FAST
def test_symbol_bindings_stay_bijective_or_raise(pairs: list[tuple[str, str]]) -> None:
    with TemporaryDirectory() as raw:
        session = _session(Path(raw))
        for internal, broker_symbol in pairs:
            with suppress(LiveGateError):
                session.bind_symbol(internal=internal, broker=broker_symbol, actor="ada")
            assert mapping_is_bijective(session._symbol_map)
