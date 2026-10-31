"""Fill ledger for the armed gateway.

Cash and positions here are broker accounting. They are not a research
headline and ``live_pnl_claim`` stays false.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


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
        self._fills: dict[str, LedgerFill] = {}
        self._order: list[str] = []
        self._by_order: dict[str, list[str]] = {}
        self._last_ns: dict[str, int] = {}
        self._duplicate_fills = 0
        self._orphan_fills = 0
        self._out_of_order_fills = 0
        self._declared_cash: float | None = None
        self._declared_positions: dict[str, float] | None = None

    def note_order(self, client_order_id: str, *, expects_fill: bool) -> None:
        self._expects[str(client_order_id)] = bool(expects_fill)

    def set_expects_fill(self, client_order_id: str, expects_fill: bool) -> None:
        key = str(client_order_id)
        if key not in self._expects:
            raise ValueError("order is not in the ledger")
        self._expects[key] = bool(expects_fill)

    def record_fill(self, fill: LedgerFill) -> str:
        """Apply a fill once. A later venue fill replaces an ``order:`` provisional."""
        if fill.side not in {"buy", "sell"}:
            raise ValueError("fill side must be buy or sell")
        if fill.fill_id in self._fills:
            self._duplicate_fills += 1
            return "duplicate_fill"
        replaced = self._drop_provisional(fill)
        if fill.client_order_id not in self._expects:
            self._fills[fill.fill_id] = fill
            self._order.append(fill.fill_id)
            self._orphan_fills += 1
            return "orphan_fill"
        verdict = "replaced_provisional" if replaced else "accepted"
        last = self._last_ns.get(fill.client_order_id)
        if last is not None and fill.fill_time_ns < last:
            self._out_of_order_fills += 1
            if verdict == "accepted":
                verdict = "out_of_order_fill"
        self._apply(fill)
        self._fills[fill.fill_id] = fill
        self._order.append(fill.fill_id)
        self._by_order.setdefault(fill.client_order_id, []).append(fill.fill_id)
        self._last_ns[fill.client_order_id] = max(fill.fill_time_ns, last or fill.fill_time_ns)
        return verdict

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
            drift: dict[str, Any] = {
                "declared": False,
                "cash_drift": None,
                "position_drift": {},
                "alarm": False,
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
        clean = not (self._orphan_fills or self._out_of_order_fills or missing or drift["alarm"])
        return {
            "ok": clean,
            "missing_fills": missing,
            "duplicate_fills": self._duplicate_fills,
            "orphan_fills": self._orphan_fills,
            "out_of_order_fills": self._out_of_order_fills,
            "drift": drift,
            "computed_cash": self._cash,
            "computed_positions": dict(self._positions),
            "live_pnl_claim": False,
            "research_evidence": False,
        }

    def _drop_provisional(self, fill: LedgerFill) -> bool:
        if fill.fill_id.startswith("order:"):
            return False
        doomed = [
            fill_id
            for fill_id, prior in self._fills.items()
            if prior.client_order_id == fill.client_order_id and fill_id.startswith("order:")
        ]
        if not doomed:
            return False
        for fill_id in doomed:
            prior = self._fills.pop(fill_id)
            self._order.remove(fill_id)
            bucket = self._by_order.get(prior.client_order_id, [])
            if fill_id in bucket:
                bucket.remove(fill_id)
            self._unwind(prior)
            if self._last_ns.get(prior.client_order_id) == prior.fill_time_ns:
                remaining = self._by_order.get(prior.client_order_id, [])
                if remaining:
                    self._last_ns[prior.client_order_id] = max(
                        self._fills[item].fill_time_ns for item in remaining
                    )
                else:
                    self._last_ns.pop(prior.client_order_id, None)
        return True

    def _apply(self, fill: LedgerFill) -> None:
        signed = fill.quantity if fill.side == "buy" else -fill.quantity
        self._positions[fill.symbol] = self._positions.get(fill.symbol, 0.0) + signed
        self._cash -= signed * fill.price

    def _unwind(self, fill: LedgerFill) -> None:
        signed = fill.quantity if fill.side == "buy" else -fill.quantity
        self._positions[fill.symbol] = self._positions.get(fill.symbol, 0.0) - signed
        self._cash += signed * fill.price
