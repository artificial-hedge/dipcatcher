"""One synthesized NYSE day through the offline gateway. SYNTHETIC.

Open auction, a halt, an LULD breach, a sequence gap, and the close are
denied. A clean in-session quote is the only print that authorizes an order.
The JSON summary is the artifact for this replay.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_fund.live.exchange_clock import nyse_rth_closed
from tests.unit.fault.support import FaultBroker, arm, buy, open_fault_session

_ET = ZoneInfo("America/New_York")
_DAY = date(2024, 1, 3)


def _ns(hour: int, minute: int, second: int = 0) -> int:
    stamp = datetime(_DAY.year, _DAY.month, _DAY.day, hour, minute, second, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_synthesized_trading_day_blocks_stale_and_invalid_data(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path / "day", broker, now=_ns(9, 0))
    arm(session)
    session.bind_symbol(internal="AAPL", broker="AAPL", actor="ada")
    events: list[dict[str, object]] = []
    submits = 0

    def record(phase: str, tradable: bool, stamp: int, **fields: object) -> None:
        events.append(
            {
                "phase": phase,
                "tradable": tradable,
                "ts_ns": stamp,
                "market_closed": nyse_rth_closed(stamp),
                **fields,
            }
        )

    pre = _ns(9, 0)
    closed_quote = session.ingest_quote(
        symbol="AAPL", bid=100.0, ask=100.1, seq=1, actor="ada", at_ns=pre
    )
    pre_order = buy(session, client_order_id="pre", at_ns=pre + 1, mark=100.0)
    record(
        "pre_open",
        False,
        pre,
        quote_ok=closed_quote["ok"],
        authorized=bool(pre_order["ok"]),
        reasons=list(pre_order["reasons"]),
    )
    assert closed_quote["reasons"] == ["market_closed"]
    assert "AAPL" not in session.marks

    auction = _ns(9, 30, 1)
    crossed = session.ingest_quote(
        symbol="AAPL", bid=101.0, ask=100.0, seq=1, actor="ada", at_ns=auction
    )
    auction_order = buy(session, client_order_id="auction", at_ns=auction + 1, mark=101.0)
    record(
        "open_auction",
        False,
        auction,
        quote_ok=crossed["ok"],
        quote_reasons=list(crossed["reasons"]),
        authorized=bool(auction_order["ok"]),
        reasons=list(auction_order["reasons"]),
    )
    assert "crossed" in crossed["reasons"]
    assert "AAPL" not in session.marks
    assert auction_order["reasons"] == ["invalid_quote"]

    regular = _ns(9, 31)
    good = session.ingest_quote(
        symbol="AAPL", bid=100.0, ask=100.1, seq=1, actor="ada", at_ns=regular
    )
    opened = buy(
        session,
        client_order_id="open",
        at_ns=regular + 1,
        mark=100.05,
        limit_price=100.05,
    )
    record(
        "regular_open",
        True,
        regular,
        quote_ok=good["ok"],
        authorized=bool(opened["ok"]),
        reasons=list(opened.get("reasons", [])),
        mark=session.marks.get("AAPL"),
    )

    halt_at = regular + 2_100_000_000
    session.ingest_quote(symbol="AAPL", bid=100.0, ask=100.1, seq=2, actor="ada", at_ns=halt_at)
    session.set_symbol_controls(symbol="AAPL", actor="ada", halted=True, at_ns=halt_at + 1)
    halted = buy(
        session, client_order_id="halt", at_ns=halt_at + 2, mark=100.05, limit_price=100.05
    )
    record(
        "halt",
        False,
        halt_at,
        authorized=bool(halted["ok"]),
        reasons=list(halted["reasons"]),
    )
    session.set_symbol_controls(symbol="AAPL", actor="ada", halted=False, at_ns=halt_at + 3)

    band_at = halt_at + 1_000_000_000
    session.set_price_band(symbol="AAPL", lower=95.0, upper=105.0, actor="ada")
    luld_quote = session.ingest_quote(
        symbol="AAPL", bid=110.0, ask=110.2, seq=3, actor="ada", at_ns=band_at
    )
    luld_order = buy(
        session, client_order_id="luld", at_ns=band_at + 1, mark=110.1, limit_price=110.1
    )
    record(
        "luld",
        False,
        band_at,
        quote_ok=luld_quote["ok"],
        quote_reasons=list(luld_quote["reasons"]),
        authorized=bool(luld_order["ok"]),
        reasons=list(luld_order["reasons"]),
        mark=session.marks["AAPL"],
    )

    gap_at = band_at + 1_000_000_000
    gap = session.ingest_quote(
        symbol="AAPL", bid=100.0, ask=100.1, seq=6, actor="ada", at_ns=gap_at
    )
    gap_order = buy(
        session, client_order_id="gap", at_ns=gap_at + 1, mark=100.05, limit_price=100.05
    )
    record(
        "sequence_gap",
        False,
        gap_at,
        quote_ok=gap["ok"],
        quote_reasons=list(gap["reasons"]),
        authorized=bool(gap_order["ok"]),
        reasons=list(gap_order["reasons"]),
    )

    resume = gap_at + 1_000_000_000
    healed = session.ingest_quote(
        symbol="AAPL", bid=100.0, ask=100.2, seq=4, actor="ada", at_ns=resume
    )
    resumed = buy(
        session,
        client_order_id="resume",
        at_ns=resume + 1,
        mark=100.1,
        limit_price=100.1,
    )
    record(
        "resume",
        True,
        resume,
        quote_ok=healed["ok"],
        authorized=bool(resumed["ok"]),
        reasons=list(resumed.get("reasons", [])),
    )

    close = _ns(16, 0)
    closed = buy(session, client_order_id="close", at_ns=close, mark=100.1, limit_price=100.1)
    record(
        "close",
        False,
        close,
        authorized=bool(closed["ok"]),
        reasons=list(closed["reasons"]),
    )

    submits = broker.submit_count
    summary = {
        "schema": "live_day_replay.v1",
        "day": _DAY.isoformat(),
        "synthetic": True,
        "live_pnl_claim": False,
        "research_evidence": False,
        "events": events,
        "broker_submits": submits,
        "authorized": [row["phase"] for row in events if row["authorized"]],
    }
    path = tmp_path / "day_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert summary["authorized"] == ["regular_open", "resume"]
    assert submits == 2
    for row in events:
        if row["tradable"]:
            assert row["authorized"] is True
        else:
            assert row["authorized"] is False
            assert row["reasons"]
    assert events[0]["reasons"] == ["market_closed"]
    assert "crossed" in events[1]["quote_reasons"]
    assert events[1]["reasons"] == ["invalid_quote"]
    assert events[3]["reasons"] == ["halted"]
    assert "luld" in events[4]["reasons"]
    assert events[4]["mark"] == 100.05
    assert events[5]["reasons"] == ["stale_data"]
    assert events[7]["reasons"] == ["market_closed"]
    assert session.marks["AAPL"] == 100.1
