"""Multi-session SYNTHETIC paper soak.

Thirty or more sessions of the synthetic book go through the simulated
broker. After every session the soak checks cash conservation, position
reconciliation, and the mark-to-fill accounting identity. It is an
accounting check, labeled SYNTHETIC, and it is not market evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.paper.trading_session import SessionLimits, run_session
from quant_fund.pipeline.e2e_smoke import require_finite, require_populated, require_rows
from quant_fund.simtest.e2e_paper import E2EPaperError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.seeds import set_global_seed

_MIN_SESSIONS = 30
_NAMES = 2
_QTY = 1.0
_STARTING_CASH = 1_000_000.0
_CASH_EPS = 1e-6
_QTY_EPS = 1e-9
_PANEL_ASSETS = 4
_OK = {"mark": "marked", "submit": "accepted", "fill": "accepted"}


class E2ESoakError(E2EPaperError):
    """The paper soak broke an accounting identity or a session contract."""


def run_paper_soak(
    root: str | Path,
    *,
    sessions: int = _MIN_SESSIONS,
    seed: int = 13,
) -> dict[str, Any]:
    """Trade ``sessions`` synthetic days and return a digest-bearing receipt."""
    count = _require_sessions(sessions)
    _require_seed(seed)
    lake = Path(root)
    lake.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    selected = select_soak_sessions(_bars(count, seed), count)
    limits = _limits()
    state = lake / "metadata" / "paper_soak.json"
    book = _Book()
    for index, day in enumerate(selected):
        report = run_session(
            state_path=state,
            commands=_day_commands(index, day),
            resume=index > 0,
            starting_cash=_STARTING_CASH,
            limits=limits,
        )
        book.observe(index, day, report)
    return _receipt(seed, count, selected, book)


def select_soak_sessions(bars: pl.DataFrame, sessions: int) -> list[dict[str, Any]]:
    """Last ``sessions`` dates and the first two names present on every date."""
    count = _require_sessions(sessions)
    _require_book(bars)
    times = bars.get_column("event_time").unique().sort()
    if times.len() < count:
        raise E2ESoakError(f"paper-soak: fewer than {count} sessions")
    chosen = times.tail(count).to_list()
    if any(value is None for value in chosen):
        raise E2ESoakError("paper-soak: empty session time")
    names = _names_for(bars, chosen)
    return [_day(bars, when, names) for when in chosen]


def _bars(sessions: int, seed: int) -> pl.DataFrame:
    # The delisted name stops before the sample end, so the panel needs
    # extra sessions for two names to cover the soak window.
    provider = SyntheticMarketProvider(
        n_assets=_PANEL_ASSETS,
        n_days=sessions + 16,
        seed=seed,
    )
    return provider.get_bars()


def _require_sessions(sessions: int) -> int:
    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < _MIN_SESSIONS:
        raise E2ESoakError(f"paper-soak: sessions must be an int >= {_MIN_SESSIONS}")
    return sessions


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise E2ESoakError("paper-soak: seed must be an int in [0, 2**32)")


def _require_book(bars: pl.DataFrame) -> None:
    require_rows("paper-soak", bars)
    require_populated("paper-soak", bars, ("security_id", "event_time", "close", "source"))
    sources = {str(value) for value in bars.get_column("source").unique().to_list()}
    if sources != {"synthetic"}:
        raise E2ESoakError("paper-soak: source is not synthetic")


def _names_for(bars: pl.DataFrame, times: Sequence[object]) -> list[str]:
    window = bars.filter(pl.col("event_time").is_in(list(times)))
    counts = window.group_by("security_id").len().sort("security_id")
    full = counts.filter(pl.col("len") == len(times))
    if full.height < _NAMES:
        raise E2ESoakError("paper-soak: fewer than 2 complete names")
    return [str(value) for value in full.get_column("security_id").head(_NAMES).to_list()]


def _day(bars: pl.DataFrame, when: object, names: Sequence[str]) -> dict[str, Any]:
    return {"session_time": _as_stamp(when), "quotes": [_quote(bars, when, name) for name in names]}


def _as_stamp(when: object) -> str:
    isoformat = getattr(when, "isoformat", None)
    if not callable(isoformat):
        raise E2ESoakError("paper-soak: session time is empty")
    stamp = str(isoformat())
    if not stamp:
        raise E2ESoakError("paper-soak: session time is empty")
    return stamp


def _quote(bars: pl.DataFrame, when: object, name: str) -> dict[str, Any]:
    rows = bars.filter((pl.col("security_id") == name) & (pl.col("event_time") == when))
    if rows.height != 1:
        raise E2ESoakError(f"paper-soak: expected one bar for {name}")
    price = float(rows.get_column("close")[0])
    if not math.isfinite(price) or price <= 0.0:
        raise E2ESoakError(f"paper-soak: non-finite close for {name}")
    return {"security_id": name, "price": price}


def _limits() -> SessionLimits:
    return SessionLimits(
        max_order_quantity=10.0,
        max_order_notional=1_000_000.0,
        max_position_notional=5_000_000.0,
        max_gross_notional=5_000_000.0,
    )


def _day_commands(index: int, day: Mapping[str, Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    when = str(day["session_time"])
    for quote in day["quotes"]:
        security_id = str(quote["security_id"])
        price = float(quote["price"])
        order_id = f"d{index}-{security_id}"
        commands.append({"op": "mark", "security_id": security_id, "price": price})
        commands.append(
            {
                "op": "submit",
                "order_id": order_id,
                "security_id": security_id,
                "side": "BUY",
                "quantity": _QTY,
            }
        )
        commands.append(
            {
                "op": "fill",
                "fill_id": f"f{index}-{security_id}",
                "order_id": order_id,
                "security_id": security_id,
                "quantity": _QTY,
                "price": price,
                "fill_time": when,
            }
        )
    if not commands:
        raise E2ESoakError("paper-soak: empty session")
    return commands


class _Book:
    """Prior broker snapshot plus the notionals this soak decided to trade."""

    def __init__(self) -> None:
        self.cash = _STARTING_CASH
        self.positions: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self.max_cash_gap = 0.0
        self.max_position_gap = 0.0
        self.max_trade_residual = 0.0
        self.max_mark_residual = 0.0
        self.days = 0

    def observe(self, index: int, day: Mapping[str, Any], report: Mapping[str, Any]) -> None:
        _require_steps(report, index)
        quotes = _quote_map(day)
        notional = _notional(quotes)
        broker_cash, broker_positions = _broker_book(report)
        trade_gap = trade_residual(self.cash, broker_cash, notional)
        mark_gap = mark_residual(
            self.cash,
            self.positions,
            self.marks,
            broker_cash,
            broker_positions,
            quotes,
        )
        self._note_gaps(index, trade_gap, mark_gap, broker_cash, broker_positions, quotes)
        self.cash = broker_cash
        self.positions = broker_positions
        self.marks = quotes
        self.days += 1

    def _note_gaps(
        self,
        index: int,
        trade_gap: float,
        mark_gap: float,
        broker_cash: float,
        broker_positions: Mapping[str, float],
        quotes: Mapping[str, float],
    ) -> None:
        self.max_trade_residual = max(self.max_trade_residual, abs(trade_gap))
        self.max_mark_residual = max(self.max_mark_residual, abs(mark_gap))
        self.max_cash_gap = max(self.max_cash_gap, abs(trade_gap))
        if abs(trade_gap) > _CASH_EPS:
            raise E2ESoakError(f"paper-soak: cash diverged on session {index}")
        if abs(mark_gap) > _CASH_EPS:
            raise E2ESoakError(f"paper-soak: mark residual on session {index}")
        expected = {name: self.positions.get(name, 0.0) + _QTY for name in quotes}
        if set(broker_positions) != set(expected):
            raise E2ESoakError(f"paper-soak: positions diverged on session {index}")
        for name, qty in expected.items():
            gap = abs(broker_positions[name] - qty)
            self.max_position_gap = max(self.max_position_gap, gap)
            if gap > _QTY_EPS:
                raise E2ESoakError(f"paper-soak: position diverged for {name}")
        if not math.isfinite(broker_cash):
            raise E2ESoakError("paper-soak: non-finite cash")


def trade_residual(cash_before: float, cash_after: float, notional: float) -> float:
    """Cash spent since the previous session, minus the filled notional."""
    if not all(math.isfinite(value) for value in (cash_before, cash_after, notional)):
        raise E2ESoakError("paper-soak: non-finite cash")
    return (cash_before - cash_after) - notional


def mark_residual(
    cash_before: float,
    positions_before: Mapping[str, float],
    marks_before: Mapping[str, float],
    cash_after: float,
    positions_after: Mapping[str, float],
    marks_after: Mapping[str, float],
) -> float:
    """Change in book value beyond the overnight mark move.

    A fill at the mark exchanges cash for shares of equal marked value, so
    the book value moves only with the marks on shares already held.
    """
    before_marks = marks_before if marks_before else marks_after
    before = _equity(cash_before, positions_before, before_marks)
    after = _equity(cash_after, positions_after, marks_after)
    return after - before - _mark_move(positions_before, marks_before, marks_after)


def _quote_map(day: Mapping[str, Any]) -> dict[str, float]:
    quotes = {str(quote["security_id"]): float(quote["price"]) for quote in day["quotes"]}
    if len(quotes) < _NAMES:
        raise E2ESoakError("paper-soak: empty session")
    for price in quotes.values():
        if not math.isfinite(price) or price <= 0.0:
            raise E2ESoakError("paper-soak: non-finite close")
    return quotes


def _notional(quotes: Mapping[str, float]) -> float:
    return _QTY * sum(quotes.values())


def _mark_move(
    positions: Mapping[str, float],
    old_marks: Mapping[str, float],
    new_marks: Mapping[str, float],
) -> float:
    total = 0.0
    for name, qty in positions.items():
        if abs(qty) <= _QTY_EPS:
            continue
        if name not in old_marks or name not in new_marks:
            raise E2ESoakError(f"paper-soak: unmarked position in {name}")
        total += qty * (new_marks[name] - old_marks[name])
    return total


def _equity(cash: float, positions: Mapping[str, float], marks: Mapping[str, float]) -> float:
    value = cash
    for name, qty in positions.items():
        if abs(qty) <= _QTY_EPS:
            continue
        price = marks.get(name)
        if price is None or not math.isfinite(price):
            raise E2ESoakError(f"paper-soak: unmarked position in {name}")
        value += qty * price
    if not math.isfinite(value):
        raise E2ESoakError("paper-soak: non-finite book value")
    return value


def _broker_book(report: Mapping[str, Any]) -> tuple[float, dict[str, float]]:
    cash = float(report["reconciliation"]["computed_cash"])
    if not math.isfinite(cash):
        raise E2ESoakError("paper-soak: non-finite cash")
    return cash, _live_positions(report)


def _live_positions(report: Mapping[str, Any]) -> dict[str, float]:
    raw = report["reconciliation"]["computed_positions"]
    if not isinstance(raw, dict) or not raw:
        raise E2ESoakError("paper-soak: empty positions")
    live: dict[str, float] = {}
    for key, value in raw.items():
        qty = float(value)
        if not math.isfinite(qty):
            raise E2ESoakError("paper-soak: non-finite positions")
        if abs(qty) > _QTY_EPS:
            live[str(key)] = qty
    if not live:
        raise E2ESoakError("paper-soak: empty positions")
    return live


def _require_steps(report: Mapping[str, Any], index: int) -> None:
    steps = report["steps"]
    if not steps:
        raise E2ESoakError(f"paper-soak: empty session {index}")
    for step in steps:
        expected = _OK.get(str(step["op"]))
        if expected is None or step["verdict"] != expected:
            raise E2ESoakError(f"paper-soak: session {index} {step['op']} {step['verdict']}")
    if not report["consistent"] or not report["nav_seam_parity"]:
        raise E2ESoakError(f"paper-soak: seam failed on session {index}")
    if report["live_pnl_claim"] is not False or report["would_promote_live"] is not False:
        raise E2ESoakError("paper-soak: live claim flag")
    if report["reconciliation"]["kill_state"] != "ENABLED":
        raise E2ESoakError(f"paper-soak: kill state on session {index}")


def _receipt(
    seed: int, sessions: int, selected: Sequence[Mapping[str, Any]], book: _Book
) -> dict[str, Any]:
    if book.days != sessions or book.days < _MIN_SESSIONS:
        raise E2ESoakError("paper-soak: not every session was checked")
    names = [str(quote["security_id"]) for quote in selected[0]["quotes"]]
    body: dict[str, Any] = {
        "schema": "e2e_paper_soak.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "live_orders_enabled": False,
        "would_promote_live": False,
        "seed": seed,
        "sessions": sessions,
        "names": names,
        "fills": sessions * len(names),
        "days_checked": book.days,
        "positions": dict(book.positions),
        "cash": book.cash,
        "max_cash_gap": book.max_cash_gap,
        "max_position_gap": book.max_position_gap,
        "max_trade_residual": book.max_trade_residual,
        "max_mark_residual": book.max_mark_residual,
    }
    require_finite("paper-soak", body)
    if not body["positions"]:
        raise E2ESoakError("paper-soak: empty positions")
    return {**body, "digest": hash_bytes(canonical_json_bytes(body))}
