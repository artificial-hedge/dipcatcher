"""Live quote, reference, and corporate-action checks.

A rejected tick does not become a mark. A sequence gap, duplicate, or
reversal marks the feed stale instead of inventing the missing print.
Corporate actions keep cash plus position value unchanged, or they halt
the name and leave the book alone. Nothing here is a research receipt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A wider print is not a mark. Equality with the band is still a real print.
MAX_SPREAD = 0.05
MAX_JUMP = 0.25


class MarketDataError(ValueError):
    """The quote, map, or action cannot be applied."""


@dataclass(frozen=True)
class QuoteDecision:
    """One sequenced quote. ``accept`` is the only path that may set a mark."""

    accept: bool
    stale: bool
    reasons: tuple[str, ...]
    mark: float | None
    seq: int | None


@dataclass(frozen=True)
class ReferenceSnapshot:
    """Versioned lot and tick sizes. A stale snapshot does not size orders."""

    version: int
    as_of_ns: int
    max_age_ns: int
    lot_size: dict[str, float]
    tick_size: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "as_of_ns": self.as_of_ns,
            "max_age_ns": self.max_age_ns,
            "lot_size": dict(self.lot_size),
            "tick_size": dict(self.tick_size),
        }


@dataclass(frozen=True)
class ActionPlan:
    """Economics to apply. ``apply`` false leaves cash, positions, and marks."""

    apply: bool
    halt: bool
    reason: str
    cash: float
    positions: dict[str, float]
    marks: dict[str, float]
    symbol: str
    new_symbol: str
    qty_factor: float
    cash_delta: float


def price_problem(price: float) -> str:
    """``bad_price`` for a non-finite, zero, or negative print."""
    if not math.isfinite(price) or price <= 0.0:
        return "bad_price"
    return ""


def jump_problem(previous: float | None, price: float, *, max_jump: float = MAX_JUMP) -> str:
    """``price_jump`` when the move is strictly outside the sanity band."""
    if previous is None or not math.isfinite(previous) or previous <= 0.0:
        return ""
    if not math.isfinite(price) or price <= 0.0:
        return "bad_price"
    if abs(price - previous) / previous > max_jump:
        return "price_jump"
    return ""


def quote_problems(
    bid: float,
    ask: float,
    *,
    previous: float | None,
    max_spread: float = MAX_SPREAD,
    max_jump: float = MAX_JUMP,
) -> tuple[str, ...]:
    """Reject a quote that must not become a mark. Empty means the mid is usable."""
    reasons: list[str] = []
    if price_problem(bid):
        reasons.append("bad_bid")
    if price_problem(ask):
        reasons.append("bad_ask")
    if reasons:
        return tuple(reasons)
    _append_book(reasons, bid, ask, max_spread)
    mid = (bid + ask) / 2.0
    jumped = jump_problem(previous, mid, max_jump=max_jump)
    if jumped:
        reasons.append(jumped)
    return tuple(reasons)


def sequence_problem(previous: int | None, seq: int, *, snapshot: bool) -> str:
    """Name a sequence fault. Empty means this print is the next one, or a snapshot."""
    if seq < 0:
        return "bad_sequence"
    if snapshot or previous is None:
        return ""
    if seq == previous:
        return "duplicate_sequence"
    if seq < previous:
        return "out_of_order"
    if seq > previous + 1:
        return "sequence_gap"
    return ""


def assess_quote(
    *,
    bid: float,
    ask: float,
    seq: int,
    previous_seq: int | None,
    previous_mark: float | None,
    was_stale: bool,
    snapshot: bool,
    max_spread: float = MAX_SPREAD,
    max_jump: float = MAX_JUMP,
) -> QuoteDecision:
    """Accept only a clean in-order quote. Sequence faults latch stale and keep the mark."""
    fault = sequence_problem(previous_seq, seq, snapshot=snapshot)
    reasons = quote_problems(
        bid, ask, previous=previous_mark, max_spread=max_spread, max_jump=max_jump
    )
    if fault:
        return QuoteDecision(False, True, (fault, *reasons), previous_mark, previous_seq)
    if reasons:
        return QuoteDecision(False, was_stale, reasons, previous_mark, previous_seq)
    mid = (bid + ask) / 2.0
    return QuoteDecision(True, False, (), mid, seq)


def snapshot_problem(snapshot: ReferenceSnapshot | None, symbol: str, ts_ns: int) -> str:
    """Empty when no snapshot is installed. A stale or unknown one fails closed."""
    if snapshot is None:
        return ""
    if snapshot.version < 1 or snapshot.max_age_ns < 0 or snapshot.as_of_ns < 0:
        return "bad_reference_version"
    if ts_ns < snapshot.as_of_ns or ts_ns > snapshot.as_of_ns + snapshot.max_age_ns:
        return "stale_reference"
    lot = snapshot.lot_size.get(symbol)
    tick = snapshot.tick_size.get(symbol)
    if lot is None or tick is None:
        return "unknown_reference"
    if price_problem(lot) or price_problem(tick):
        return "bad_reference"
    return ""


def size_problems(qty: float, price: float, lot: float, tick: float) -> tuple[str, ...]:
    """Lot and tick misses for one symbol in the active snapshot."""
    reasons: list[str] = []
    if not _on_grid(qty, lot):
        reasons.append("lot_size")
    if not _on_grid(price, tick):
        reasons.append("tick_size")
    return tuple(reasons)


def bind_symbol(mapping: dict[str, str], internal: str, broker: str) -> None:
    """Bind one internal id to one broker symbol. A second binding is an error."""
    left = _name(internal)
    right = _name(broker)
    current = mapping.get(left)
    if current is not None and current != right:
        raise MarketDataError(f"{left} is already bound to {current}")
    owner = _owner(mapping, right)
    if owner is not None and owner != left:
        raise MarketDataError(f"{right} is already bound to {owner}")
    mapping[left] = right


def require_internal(mapping: dict[str, str], broker: str) -> str:
    """Internal id for a broker symbol. Unknown symbols raise."""
    right = _name(broker)
    owner = _owner(mapping, right)
    if owner is None:
        raise MarketDataError(f"unknown broker symbol {right}")
    return owner


def mapping_is_bijective(mapping: dict[str, str]) -> bool:
    """True when every internal id and every broker symbol appears once."""
    if any(not _clean(left) or not _clean(right) for left, right in mapping.items()):
        return False
    return len(set(mapping.values())) == len(mapping)


def plan_action(
    *,
    kind: str,
    symbol: str,
    cash: float,
    positions: dict[str, float],
    marks: dict[str, float],
    working: bool,
    ratio: float | None = None,
    cash_per_share: float | None = None,
    new_symbol: str | None = None,
    qty_cap: float | None = None,
) -> ActionPlan:
    """Adjust a name, or halt it, without changing account value."""
    name = _name(symbol)
    if working:
        return _hold(name, cash, positions, marks, "working_orders")
    if kind == "split":
        return _plan_split(name, cash, positions, marks, ratio, qty_cap, reverse=False)
    if kind == "reverse_split":
        return _plan_split(name, cash, positions, marks, ratio, qty_cap, reverse=True)
    if kind == "dividend":
        return _plan_dividend(name, cash, positions, marks, cash_per_share)
    if kind == "symbol_change":
        return _plan_rename(name, cash, positions, marks, new_symbol)
    if kind == "delisting":
        return _hold(name, cash, positions, marks, "delisting")
    return _hold(name, cash, positions, marks, "unknown_action")


def equity(cash: float, positions: dict[str, float], marks: dict[str, float]) -> float:
    """Cash plus marked positions. Missing marks contribute nothing only when flat."""
    total = float(cash)
    for symbol, qty in positions.items():
        if qty == 0.0:
            continue
        mark = marks.get(symbol)
        if mark is None:
            return float("nan")
        total += float(qty) * float(mark)
    return total


def snapshot_from_dict(raw: dict[str, object]) -> ReferenceSnapshot:
    """Restore a snapshot. A malformed payload is a hard error."""
    lots = raw.get("lot_size")
    ticks = raw.get("tick_size")
    if not isinstance(lots, dict) or not isinstance(ticks, dict):
        raise MarketDataError("reference snapshot is malformed")
    return ReferenceSnapshot(
        version=_as_int(raw.get("version")),
        as_of_ns=_as_int(raw.get("as_of_ns")),
        max_age_ns=_as_int(raw.get("max_age_ns")),
        lot_size={str(key): float(value) for key, value in lots.items()},
        tick_size={str(key): float(value) for key, value in ticks.items()},
    )


def _append_book(reasons: list[str], bid: float, ask: float, max_spread: float) -> None:
    if bid > ask:
        reasons.append("crossed")
        return
    if bid == ask:
        reasons.append("locked")
        return
    mid = (bid + ask) / 2.0
    if (ask - bid) / mid > max_spread:
        reasons.append("wide_spread")


def _on_grid(value: float, step: float) -> bool:
    if not math.isfinite(value) or not math.isfinite(step) or step <= 0.0 or value <= 0.0:
        return False
    units = round(value / step)
    return abs(value - (units * step)) <= step * 1e-6


def _name(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not _clean(cleaned):
        raise MarketDataError("symbol is empty or not a ticker")
    return cleaned


def _clean(symbol: str) -> bool:
    if not symbol or len(symbol) > 16:
        return False
    return all(ch.isalnum() or ch in ".-" for ch in symbol)


def _owner(mapping: dict[str, str], broker: str) -> str | None:
    found = [left for left, right in mapping.items() if right == broker]
    if len(found) > 1:
        raise MarketDataError(f"{broker} is not bijective")
    if not found:
        return None
    return found[0]


def _hold(
    symbol: str, cash: float, positions: dict[str, float], marks: dict[str, float], reason: str
) -> ActionPlan:
    return ActionPlan(
        False, True, reason, cash, dict(positions), dict(marks), symbol, symbol, 1.0, 0.0
    )


def _plan_split(
    symbol: str,
    cash: float,
    positions: dict[str, float],
    marks: dict[str, float],
    ratio: float | None,
    qty_cap: float | None,
    *,
    reverse: bool,
) -> ActionPlan:
    if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
        return _hold(symbol, cash, positions, marks, "bad_ratio")
    if reverse and ratio >= 1.0:
        return _hold(symbol, cash, positions, marks, "bad_ratio")
    if not reverse and ratio <= 1.0:
        return _hold(symbol, cash, positions, marks, "bad_ratio")
    mark = marks.get(symbol)
    if mark is None or price_problem(mark):
        return _hold(symbol, cash, positions, marks, "missing_mark")
    qty = float(positions.get(symbol, 0.0))
    new_qty = qty * ratio
    new_mark = mark / ratio
    if price_problem(new_mark):
        return _hold(symbol, cash, positions, marks, "bad_price")
    if qty_cap is not None and abs(new_qty) > qty_cap:
        return _hold(symbol, cash, positions, marks, "position_limit")
    new_positions = dict(positions)
    new_marks = dict(marks)
    if new_qty == 0.0:
        new_positions.pop(symbol, None)
    else:
        new_positions[symbol] = new_qty
    new_marks[symbol] = new_mark
    return _guard_nav(
        symbol, symbol, cash, positions, marks, cash, new_positions, new_marks, ratio, 0.0
    )


def _plan_dividend(
    symbol: str,
    cash: float,
    positions: dict[str, float],
    marks: dict[str, float],
    cash_per_share: float | None,
) -> ActionPlan:
    if cash_per_share is None or not math.isfinite(cash_per_share) or cash_per_share <= 0.0:
        return _hold(symbol, cash, positions, marks, "bad_dividend")
    mark = marks.get(symbol)
    if mark is None or price_problem(mark):
        return _hold(symbol, cash, positions, marks, "missing_mark")
    qty = float(positions.get(symbol, 0.0))
    new_mark = mark - cash_per_share
    delta = cash_per_share * qty
    if price_problem(new_mark) or cash + delta < 0.0:
        return _hold(symbol, cash, positions, marks, "bad_price")
    new_marks = dict(marks)
    new_marks[symbol] = new_mark
    return _guard_nav(
        symbol,
        symbol,
        cash,
        positions,
        marks,
        cash + delta,
        dict(positions),
        new_marks,
        1.0,
        delta,
    )


def _plan_rename(
    symbol: str,
    cash: float,
    positions: dict[str, float],
    marks: dict[str, float],
    new_symbol: str | None,
) -> ActionPlan:
    if new_symbol is None:
        return _hold(symbol, cash, positions, marks, "bad_symbol")
    try:
        target = _name(new_symbol)
    except MarketDataError:
        return _hold(symbol, cash, positions, marks, "bad_symbol")
    if target == symbol or target in positions or target in marks:
        return _hold(symbol, cash, positions, marks, "symbol_collision")
    new_positions = dict(positions)
    new_marks = dict(marks)
    if symbol in new_positions:
        new_positions[target] = new_positions.pop(symbol)
    if symbol in new_marks:
        new_marks[target] = new_marks.pop(symbol)
    return _guard_nav(
        symbol, target, cash, positions, marks, cash, new_positions, new_marks, 1.0, 0.0
    )


def _guard_nav(
    symbol: str,
    new_symbol: str,
    cash: float,
    positions: dict[str, float],
    marks: dict[str, float],
    new_cash: float,
    new_positions: dict[str, float],
    new_marks: dict[str, float],
    qty_factor: float,
    cash_delta: float,
) -> ActionPlan:
    before = equity(cash, positions, marks)
    after = equity(new_cash, new_positions, new_marks)
    if not math.isfinite(before) or not math.isfinite(after) or abs(after - before) > 1e-4:
        return _hold(symbol, cash, positions, marks, "phantom_nav")
    return ActionPlan(
        True,
        False,
        "applied",
        new_cash,
        new_positions,
        new_marks,
        symbol,
        new_symbol,
        qty_factor,
        cash_delta,
    )


def _as_int(value: object) -> int:
    """Integer field from a snapshot payload. Booleans are not versions."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise MarketDataError("reference snapshot is malformed")
    return value
