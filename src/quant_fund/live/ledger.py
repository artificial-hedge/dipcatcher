"""Fill ledger for the armed gateway.

Cash and positions here are broker accounting. They are not a research
headline and ``live_pnl_claim`` stays false.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _require_fill(fill: LedgerFill) -> None:
    if fill.side not in {"buy", "sell"}:
        raise ValueError("fill side must be buy or sell")
    _finite_positive(fill.quantity, "fill quantity")
    _finite_positive(fill.price, "fill price")
    if not fill.fill_id or not fill.client_order_id or not fill.symbol:
        raise ValueError("fill ids and symbol are required")
    if fill.fill_time_ns < 0:
        raise ValueError("fill_time_ns must be non-negative")


def _same_fill(prior: LedgerFill, fill: LedgerFill) -> bool:
    return (
        prior.client_order_id == fill.client_order_id
        and prior.symbol == fill.symbol
        and prior.side == fill.side
        and abs(prior.quantity - fill.quantity) <= 1e-9
        and abs(prior.price - fill.price) <= 1e-9
        and prior.fill_time_ns == fill.fill_time_ns
    )


def _grows_provisional(prior: LedgerFill, fill: LedgerFill) -> bool:
    if not fill.fill_id.startswith("order:"):
        return False
    if prior.client_order_id != fill.client_order_id or prior.symbol != fill.symbol:
        return False
    if prior.side != fill.side:
        return False
    return fill.quantity > prior.quantity + 1e-9


def _finite_positive(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


@dataclass(frozen=True)
class LedgerFill:
    fill_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fill_time_ns: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "fill_time_ns": self.fill_time_ns,
        }


def ledger_fill_from_dict(raw: dict[str, Any]) -> LedgerFill:
    side = str(raw["side"])
    if side not in {"buy", "sell"}:
        raise ValueError("fill side must be buy or sell")
    fill_time = int(raw["fill_time_ns"])
    if fill_time < 0:
        raise ValueError("fill_time_ns must be non-negative")
    return LedgerFill(
        fill_id=str(raw["fill_id"]),
        client_order_id=str(raw["client_order_id"]),
        symbol=str(raw["symbol"]),
        side=side,
        quantity=_finite_positive(float(raw["quantity"]), "fill quantity"),
        price=_finite_positive(float(raw["price"]), "fill price"),
        fill_time_ns=fill_time,
    )


def _is_provisional(fill_id: str) -> bool:
    return fill_id.startswith("order:") and fill_id.endswith(":cum") and fill_id != "order::cum"


class LiveLedger:
    """Idempotent order/fill accounting with a broker-statement drift alarm."""

    def __init__(
        self,
        *,
        baseline_cash: float,
        baseline_positions: dict[str, float] | None = None,
        cash_tolerance: float = 1e-6,
        position_tolerance: float = 1e-6,
    ) -> None:
        cash = float(baseline_cash)
        if not math.isfinite(cash) or cash < 0.0:
            raise ValueError("baseline cash must be finite and non-negative")
        self.baseline_cash = cash
        self.baseline_positions = {
            str(symbol): float(qty) for symbol, qty in (baseline_positions or {}).items()
        }
        for qty in self.baseline_positions.values():
            if not math.isfinite(qty):
                raise ValueError("baseline position must be finite")
        self._cash_tolerance = float(cash_tolerance)
        self._position_tolerance = float(position_tolerance)
        self._cash = cash
        self._positions = dict(self.baseline_positions)
        self._expects: dict[str, bool] = {}
        self._authorized: dict[str, float] = {}
        self._filled_qty: dict[str, float] = {}
        self._fills: dict[str, LedgerFill] = {}
        self._applied: set[str] = set()
        self._order: list[str] = []
        self._by_order: dict[str, list[str]] = {}
        self._last_ns: dict[str, int] = {}
        self._duplicate_fills = 0
        self._conflicting_fills = 0
        self._orphan_fills = 0
        self._out_of_order_fills = 0
        self._overfills = 0
        self._declared_cash: float | None = None
        self._declared_positions: dict[str, float] | None = None
        self._superseded: dict[str, list[LedgerFill]] = {}

    def note_order(
        self,
        client_order_id: str,
        *,
        expects_fill: bool,
        quantity: float | None = None,
    ) -> None:
        """Remember an order. ``quantity`` caps applied fills when the caller knows it."""
        key = str(client_order_id)
        self._expects[key] = bool(expects_fill)
        if quantity is None:
            return
        qty = float(quantity)
        if not math.isfinite(qty) or qty <= 0.0:
            raise ValueError("order quantity must be finite and positive")
        self._authorized[key] = qty

    def set_expects_fill(self, client_order_id: str, expects_fill: bool) -> None:
        key = str(client_order_id)
        if key not in self._expects:
            raise ValueError("order is not in the ledger")
        self._expects[key] = bool(expects_fill)

    def record_fill(self, fill: LedgerFill) -> str:
        """Apply a fill once. A smaller cumulative print does not restate the book."""
        fill = ledger_fill_from_dict(fill.as_dict())
        _require_fill(fill)
        replay_verdict = self._replay_verdict(fill)
        if replay_verdict is not None:
            return replay_verdict
        if self._restates_provisional(fill):
            raise ValueError("fill id conflicts with a different execution")
        late_verdict = self._late_provisional_verdict(fill)
        if late_verdict is not None:
            return late_verdict
        replaced = self._drop_provisional(fill)
        if fill.client_order_id not in self._expects:
            self._store(fill, applied=False)
            if not replaced:
                self._orphan_fills += 1
            return "orphan_fill"
        if self._exceeds_authorization(fill):
            self._store(fill, applied=False)
            self._overfills += 1
            return "overfill"
        verdict = "replaced_provisional" if replaced else "accepted"
        last = self._last_ns.get(fill.client_order_id)
        if last is not None and fill.fill_time_ns < last:
            self._out_of_order_fills += 1
            if verdict == "accepted":
                verdict = "out_of_order_fill"
        self._apply(fill)
        self._store(fill, applied=True)
        self._by_order.setdefault(fill.client_order_id, []).append(fill.fill_id)
        self._last_ns[fill.client_order_id] = max(fill.fill_time_ns, last or fill.fill_time_ns)
        return verdict

    def _restates_provisional(self, fill: LedgerFill) -> bool:
        """A cumulative id may grow. A smaller or equal reprint is a conflict."""
        prior = self._fills.get(fill.fill_id)
        if prior is None or not fill.fill_id.startswith("order:"):
            return False
        same_order = (
            prior.client_order_id == fill.client_order_id
            and prior.symbol == fill.symbol
            and prior.side == fill.side
        )
        if not same_order:
            return False
        return fill.quantity <= prior.quantity + 1e-9

    def _replay_verdict(self, fill: LedgerFill) -> str | None:
        superseded = self._superseded.get(fill.fill_id, [])
        prior = self._fills.get(fill.fill_id) or (superseded[-1] if superseded else None)
        if prior is None:
            return None
        if prior == fill or any(previous == fill for previous in superseded):
            self._duplicate_fills += 1
            return "duplicate_fill"
        cumulative_update = _is_provisional(fill.fill_id) and (
            prior.client_order_id,
            prior.symbol,
            prior.side,
        ) == (fill.client_order_id, fill.symbol, fill.side)
        if not cumulative_update:
            self._conflicting_fills += 1
            return "conflicting_fill"
        if fill.fill_id not in self._fills or fill.fill_time_ns < prior.fill_time_ns:
            self._duplicate_fills += 1
            return "duplicate_fill"
        if fill.fill_time_ns == prior.fill_time_ns:
            self._conflicting_fills += 1
            return "conflicting_fill"
        return None

    def _late_provisional_verdict(self, fill: LedgerFill) -> str | None:
        if not _is_provisional(fill.fill_id):
            return None
        real_fills = [
            existing
            for fill_id, existing in self._fills.items()
            if existing.client_order_id == fill.client_order_id and not _is_provisional(fill_id)
        ]
        if not real_fills:
            return None
        if any(
            (existing.symbol, existing.side) != (fill.symbol, fill.side) for existing in real_fills
        ):
            self._conflicting_fills += 1
            return "conflicting_fill"
        self._superseded.setdefault(fill.fill_id, []).append(fill)
        self._duplicate_fills += 1
        return "duplicate_fill"

    def declare(self, *, cash: float, positions: dict[str, float]) -> None:
        number = float(cash)
        if not math.isfinite(number):
            raise ValueError("declared cash must be finite")
        self._declared_cash = number
        self._declared_positions = {str(k): float(v) for k, v in positions.items()}
        for qty in self._declared_positions.values():
            if not math.isfinite(qty):
                raise ValueError("declared position must be finite")

    def fills(self) -> list[LedgerFill]:
        return [self._fills[fill_id] for fill_id in self._order]

    def has_fill(self, fill_id: str) -> bool:
        return str(fill_id) in self._fills

    def was_applied(self, fill_id: str) -> bool:
        """True when the fill moved cash and positions. Orphans and overfills do not."""
        return str(fill_id) in self._applied

    @property
    def computed_cash(self) -> float:
        return self._cash

    @property
    def computed_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def reconcile(self) -> dict[str, Any]:
        missing = sorted(
            order_id
            for order_id, expects in self._expects.items()
            if expects and not self._by_order.get(order_id)
        )
        if self._declared_cash is None or self._declared_positions is None:
            # No broker statement was supplied. That is not a flat, clean book.
            drift: dict[str, Any] = {
                "declared": False,
                "cash_drift": None,
                "position_drift": {},
                "alarm": True,
            }
        else:
            cash_drift = abs(self._cash - self._declared_cash)
            symbols = set(self._positions) | set(self._declared_positions)
            position_drift = {
                symbol: abs(
                    self._positions.get(symbol, 0.0) - self._declared_positions.get(symbol, 0.0)
                )
                for symbol in sorted(symbols)
            }
            alarm = cash_drift > self._cash_tolerance or any(
                delta > self._position_tolerance for delta in position_drift.values()
            )
            drift = {
                "declared": True,
                "cash_drift": cash_drift,
                "position_drift": position_drift,
                "alarm": bool(alarm),
            }
        # Idempotent re-delivery increments ``duplicate_fills`` and does not
        # by itself make the book unclean. A caller that sees the same id
        # twice in one batch can still read the counter.
        clean = not (
            self._conflicting_fills
            or self._orphan_fills
            or self._out_of_order_fills
            or self._overfills
            or missing
            or drift["alarm"]
        )
        return {
            "ok": clean,
            "missing_fills": missing,
            "duplicate_fills": self._duplicate_fills,
            "conflicting_fills": self._conflicting_fills,
            "orphan_fills": self._orphan_fills,
            "out_of_order_fills": self._out_of_order_fills,
            "overfills": self._overfills,
            "drift": drift,
            "computed_cash": self._cash,
            "computed_positions": dict(self._positions),
            "live_pnl_claim": False,
            "research_evidence": False,
        }

    def _replay_fill(self, fill: LedgerFill) -> str:
        prior = self._fills[fill.fill_id]
        if _same_fill(prior, fill):
            self._duplicate_fills += 1
            return "duplicate_fill"
        if _grows_provisional(prior, fill):
            return self._replace_provisional(prior, fill)
        raise ValueError("fill id conflicts with a different execution")

    def _replace_provisional(self, prior: LedgerFill, fill: LedgerFill) -> str:
        """Replace a cumulative print. Unapplied orphans never move cash."""
        self._fills[fill.fill_id] = fill
        applied = fill.fill_id in self._by_order.get(fill.client_order_id, [])
        if not applied:
            return "orphan_fill"
        self._unwind(prior)
        self._apply(fill)
        self._last_ns[fill.client_order_id] = max(
            fill.fill_time_ns,
            self._last_ns.get(fill.client_order_id, fill.fill_time_ns),
        )
        return "replaced_provisional"

    def _forget_provisional(self, fill_id: str) -> None:
        """Remove one provisional print and unwind its book effect."""
        prior = self._fills.pop(fill_id)
        self._superseded.setdefault(fill_id, []).append(prior)
        self._order.remove(fill_id)
        self._applied.discard(fill_id)
        bucket = self._by_order.get(prior.client_order_id, [])
        if fill_id in bucket:
            bucket.remove(fill_id)
        if prior.client_order_id in self._expects:
            self._unwind(prior)
        if self._last_ns.get(prior.client_order_id) != prior.fill_time_ns:
            return
        remaining = self._by_order.get(prior.client_order_id, [])
        if remaining:
            self._last_ns[prior.client_order_id] = max(
                self._fills[item].fill_time_ns for item in remaining
            )
            return
        self._last_ns.pop(prior.client_order_id, None)

    def _drop_provisional(self, fill: LedgerFill) -> bool:
        """Drop cumulative prints. A venue fill also drops older ``order:`` prints."""
        incoming_is_bare = fill.fill_id.startswith("order:") and not fill.fill_id.endswith(":cum")
        doomed = []
        for fill_id, prior in self._fills.items():
            if prior.client_order_id != fill.client_order_id or not fill_id.startswith("order:"):
                continue
            cumulative = fill_id.endswith(":cum") and fill_id != "order::cum"
            if cumulative or not incoming_is_bare:
                doomed.append(fill_id)
        if not doomed:
            return False
        for fill_id in doomed:
            self._forget_provisional(fill_id)
        return True

    def _exceeds_authorization(self, fill: LedgerFill) -> bool:
        cap = self._authorized.get(fill.client_order_id)
        if cap is None:
            return False
        filled = self._filled_qty.get(fill.client_order_id, 0.0)
        return fill.quantity > (cap - filled) + 1e-9

    def _store(self, fill: LedgerFill, *, applied: bool) -> None:
        self._fills[fill.fill_id] = fill
        self._order.append(fill.fill_id)
        if applied:
            self._applied.add(fill.fill_id)

    def _apply(self, fill: LedgerFill) -> None:
        signed = fill.quantity if fill.side == "buy" else -fill.quantity
        self._positions[fill.symbol] = self._positions.get(fill.symbol, 0.0) + signed
        self._cash -= signed * fill.price
        self._filled_qty[fill.client_order_id] = (
            self._filled_qty.get(fill.client_order_id, 0.0) + fill.quantity
        )

    def _unwind(self, fill: LedgerFill) -> None:
        signed = fill.quantity if fill.side == "buy" else -fill.quantity
        self._positions[fill.symbol] = self._positions.get(fill.symbol, 0.0) - signed
        self._cash += signed * fill.price
        self._filled_qty[fill.client_order_id] = (
            self._filled_qty.get(fill.client_order_id, 0.0) - fill.quantity
        )
