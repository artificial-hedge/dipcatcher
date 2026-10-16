"""Crash-resumable PAPER/SIMULATED broker adapter — no live orders, ever.

This adapter wraps :class:`~quant_fund.execution.order_recon.OrderFillLedger`
with a durable checkpoint so a run can be killed mid-flight and resumed with:

* **NAV seam parity** — the net asset value reconstructed after resume matches
  the value persisted at the crash seam (within tolerance).
* **No duplicate orders** — order intake is idempotent by ``order_id``; a
  replayed submission is counted as a duplicate and never double-counted.

It is a simulation only. Configuring a live / production / real-money endpoint
RAISES :class:`LiveEndpointRefused` (the hard no-live guarantee). Every report
carries ``live_pnl_claim=False`` and ``research_only=True``. No live
connectivity, live fills, or live profitability are claimed or possible here.

SYNTHETIC prices/fills in tests are correctness fixtures, never market
evidence. NAV is a simulated mark-to-model accounting identity, not live P&L.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any, NoReturn, cast

from quant_fund.execution.order_recon import (
    Fill,
    LiveEndpointRefused,
    Order,
    OrderFillLedger,
    OrderLedgerError,
    refuse_live_endpoint,
)
from quant_fund.utils.atomicio import atomic_write_text


def _fill_to_dict(fill: Fill) -> dict[str, Any]:
    return {
        "fill_id": fill.fill_id,
        "order_id": fill.order_id,
        "security_id": fill.security_id,
        "quantity": fill.quantity,
        "price": fill.price,
        "fill_time": fill.fill_time.isoformat(),
        "seq": fill.seq,
    }


def _fill_from_dict(raw: dict[str, Any]) -> Fill:
    raw = _required_fields(
        raw, "fill", ("fill_id", "order_id", "security_id", "quantity", "price", "fill_time")
    )
    return Fill(
        fill_id=raw["fill_id"],
        order_id=raw["order_id"],
        security_id=raw["security_id"],
        quantity=raw["quantity"],
        price=raw["price"],
        fill_time=_parse_time(raw["fill_time"], "fill_time"),
        seq=raw.get("seq"),
    )


def _order_to_dict(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "security_id": order.security_id,
        "side": order.side,
        "quantity": order.quantity,
        "submitted_time": order.submitted_time.isoformat() if order.submitted_time else None,
        "expects_fill": order.expects_fill,
    }


def _order_from_dict(raw: dict[str, Any]) -> Order:
    raw = _required_fields(raw, "order", ("order_id", "security_id", "side", "quantity"))
    submitted = raw.get("submitted_time")
    return Order(
        order_id=raw["order_id"],
        security_id=raw["security_id"],
        side=raw["side"],
        quantity=raw["quantity"],
        submitted_time=_parse_time(submitted, "submitted_time") if submitted is not None else None,
        expects_fill=raw.get("expects_fill", True),
    )


def _required_fields(raw: object, kind: str, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(raw, dict) or any(field not in raw for field in fields):
        raise OrderLedgerError(f"checkpoint {kind} is not an object with required fields")
    return cast(dict[str, Any], raw)


def _parse_time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise OrderLedgerError(f"checkpoint {name} must be an ISO datetime string")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise OrderLedgerError(f"checkpoint {name} is not an ISO datetime") from exc


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OrderLedgerError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise OrderLedgerError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise OrderLedgerError(f"{name} must be a finite number")
    return number


def _nonnegative_finite(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0:
        raise OrderLedgerError(f"{name} must be nonnegative")
    return number


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise OrderLedgerError(f"non-finite broker state number: {value}")


@dataclass
class SeamReport:
    """Result of a crash-resume reconstruction at the seam."""

    ok: bool
    nav_seam_parity: bool
    persisted_nav: float
    reconstructed_nav: float
    nav_delta: float
    duplicate_orders: int
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "nav_seam_parity": self.nav_seam_parity,
            "persisted_nav": self.persisted_nav,
            "reconstructed_nav": self.reconstructed_nav,
            "nav_delta": self.nav_delta,
            "duplicate_orders": self.duplicate_orders,
            "errors": list(self.errors),
            "research_only": True,
            "live_pnl_claim": False,
        }


class PaperBrokerAdapter:
    """Durable, idempotent, paper-only broker with crash-resume + NAV seam."""

    def __init__(
        self,
        *,
        state_path: str | Path,
        endpoint_kind: str = "paper",
        starting_cash: float = 0.0,
        nav_tolerance: float = 1e-6,
    ) -> None:
        self.endpoint_kind = refuse_live_endpoint(endpoint_kind)
        self.state_path = Path(state_path)
        self._nav_tolerance = _nonnegative_finite(nav_tolerance, "nav_tolerance")
        self._ledger = OrderFillLedger(
            endpoint_kind=self.endpoint_kind, starting_cash=starting_cash
        )
        self._starting_cash = float(starting_cash)
        self._orders: dict[str, Order] = {}
        self._order_sequence: list[str] = []
        self._fills: dict[str, Fill] = {}
        self._marks: dict[str, float] = {}
        self._persisted_nav: float | None = None

    # -- NAV identity (simulated mark-to-model, never live P&L) --

    @property
    def nav(self) -> float:
        state = self._ledger.reconcile()
        cash = float(state["computed_cash"])
        positions = state["computed_positions"]
        value = cash + sum(float(positions[sec]) * self._marks.get(sec, 0.0) for sec in positions)
        if not isfinite(value):
            raise OrderLedgerError("computed paper NAV must be finite")
        return value

    def mark(self, security_id: str, price: float) -> None:
        if not isinstance(security_id, str) or not security_id.strip():
            raise OrderLedgerError("mark security_id must be a non-empty string")
        validated_price = _nonnegative_finite(price, "mark price")
        previous = self._marks.get(security_id)
        self._marks[security_id] = validated_price
        try:
            _ = self.nav
        except OrderLedgerError:
            if previous is None:
                del self._marks[security_id]
            else:
                self._marks[security_id] = previous
            raise

    # -- intake (idempotent) --

    def submit(self, order: Order) -> str:
        verdict = self._ledger.submit(order)
        if verdict == "accepted":
            self._orders[order.order_id] = order
            self._order_sequence.append(order.order_id)
        return verdict

    def apply_fill(self, fill: Fill) -> str:
        verdict = self._ledger.record_fill(fill)
        if verdict in ("accepted", "out_of_order_fill", "orphan_fill"):
            self._fills[fill.fill_id] = fill
        return verdict

    def trip_kill(self, reason: str) -> str:
        return self._ledger.trip_kill(reason)

    def clear_kill(self) -> str:
        return self._ledger.clear_kill()

    # -- durability --

    def _state_dict(self) -> dict[str, Any]:
        return {
            "schema": "paper_broker_state.v1",
            "endpoint_kind": self.endpoint_kind,
            "starting_cash": self._starting_cash,
            "nav_tolerance": self._nav_tolerance,
            "orders": [_order_to_dict(self._orders[oid]) for oid in self._order_sequence],
            "order_sequence": list(self._order_sequence),
            "fills": [_fill_to_dict(f) for f in self._fills.values()],
            "marks": dict(self._marks),
            "kill_state": self._ledger.kill_state,
            "nav": self.nav,
            "research_only": True,
            "live_pnl_claim": False,
        }

    def checkpoint(self) -> dict[str, Any]:
        """Persist full state atomically; returns the seam report."""
        state = self._state_dict()
        payload = json.dumps(state, sort_keys=True, allow_nan=False) + "\n"
        atomic_write_text(self.state_path, payload)
        self._persisted_nav = float(state["nav"])
        return self.seam_report()

    def seam_report(self) -> dict[str, Any]:
        """Reconstruct from the durable state and check NAV seam parity."""
        persisted = self._read_state()
        errors: list[str] = []
        if persisted.get("endpoint_kind") != self.endpoint_kind:
            errors.append("endpoint_kind_mismatch")
        replayed = self._replay(persisted)
        duplicate_orders = replayed["duplicate_orders"]
        persisted_nav = float(persisted.get("nav", 0.0))
        reconstructed_nav = replayed["nav"]
        nav_delta = abs(reconstructed_nav - persisted_nav)
        parity = nav_delta <= self._nav_tolerance
        if not parity:
            errors.append("nav_seam_mismatch")
        if duplicate_orders:
            errors.append("duplicate_orders")
        counters = replayed["ledger"]["counters"]
        for anomaly in ("duplicate_fills", "orphan_fills", "out_of_order_fills"):
            if counters[anomaly]:
                errors.append(anomaly)
        report = SeamReport(
            ok=not errors,
            nav_seam_parity=parity,
            persisted_nav=persisted_nav,
            reconstructed_nav=reconstructed_nav,
            nav_delta=nav_delta,
            duplicate_orders=duplicate_orders,
            errors=errors,
        )
        return report.as_dict()

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise OrderLedgerError(f"no broker state at {self.state_path}")
        data = json.loads(
            self.state_path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
        if not isinstance(data, dict):
            raise OrderLedgerError(f"broker state at {self.state_path} is not an object")
        if (
            data.get("schema") != "paper_broker_state.v1"
            or data.get("research_only") is not True
            or data.get("live_pnl_claim") is not False
            or data.get("kill_state") not in ("ENABLED", "HALT_NEW_ORDERS")
            or data.get("endpoint_kind") not in ("paper", "simulated")
            or not isinstance(data.get("orders"), list)
            or not isinstance(data.get("fills"), list)
            or not isinstance(data.get("marks"), dict)
        ):
            raise OrderLedgerError("invalid or non-research paper broker checkpoint")
        for name in ("starting_cash", "nav_tolerance", "nav"):
            _finite_number(data.get(name), f"checkpoint {name}")
        _nonnegative_finite(data["nav_tolerance"], "checkpoint nav_tolerance")
        for security_id, price in data["marks"].items():
            if not isinstance(security_id, str) or not security_id.strip():
                raise OrderLedgerError("checkpoint mark security_id must be non-empty")
            _nonnegative_finite(price, "checkpoint mark price")
        return cast(dict[str, Any], data)

    def _replay(self, persisted: dict[str, Any]) -> dict[str, Any]:
        ledger = OrderFillLedger(
            endpoint_kind=str(persisted.get("endpoint_kind", "paper")),
            starting_cash=float(persisted.get("starting_cash", 0.0)),
        )
        duplicate_orders = 0
        # Replay historical orders/fills BEFORE applying the persisted kill
        # state: reconstruction is not a new order, so it is not kill-guarded.
        for raw in persisted.get("orders", []):
            if ledger.submit(_order_from_dict(raw)) == "duplicate_order":
                duplicate_orders += 1
        for raw in persisted.get("fills", []):
            ledger.record_fill(_fill_from_dict(raw))
        if persisted.get("kill_state") == "HALT_NEW_ORDERS":
            ledger.trip_kill("resumed_halt")
        marks = persisted["marks"]
        state = ledger.reconcile()
        cash = float(state["computed_cash"])
        nav = cash + sum(
            float(state["computed_positions"][sec]) * marks.get(sec, 0.0)
            for sec in state["computed_positions"]
        )
        if not isfinite(nav):
            raise OrderLedgerError("replayed paper NAV must be finite")
        return {"nav": nav, "duplicate_orders": duplicate_orders, "ledger": state}

    @classmethod
    def resume(
        cls,
        state_path: str | Path,
        *,
        endpoint_kind: str = "paper",
    ) -> tuple[PaperBrokerAdapter, dict[str, Any]]:
        """Rebuild an adapter from a checkpoint and report the seam.

        Returns ``(adapter, seam_report)``. The adapter is reconstructed by
        replaying the durable orders/fills so NAV seam parity and the
        no-duplicate-order guarantee are re-checked on every resume.
        """
        refuse_live_endpoint(endpoint_kind)
        persisted_path = Path(state_path)
        probe = cls(state_path=persisted_path, endpoint_kind=endpoint_kind)
        persisted = probe._read_state()
        if persisted["endpoint_kind"] != probe.endpoint_kind:
            raise OrderLedgerError("checkpoint endpoint_kind differs from requested endpoint")
        adapter = cls(
            state_path=persisted_path,
            endpoint_kind=str(persisted.get("endpoint_kind", endpoint_kind)),
            starting_cash=float(persisted.get("starting_cash", 0.0)),
            nav_tolerance=float(persisted.get("nav_tolerance", 1e-6)),
        )
        for raw in persisted.get("orders", []):
            order = _order_from_dict(raw)
            adapter.submit(order)
        for raw in persisted.get("fills", []):
            adapter.apply_fill(_fill_from_dict(raw))
        for sec, price in persisted["marks"].items():
            adapter.mark(sec, price)
        # A checkpoint is locally editable, so a resumed adapter always starts
        # halted. An operator must explicitly clear it after reviewing the seam.
        reason = (
            "resumed_halt"
            if persisted.get("kill_state") == "HALT_NEW_ORDERS"
            else "resume_requires_explicit_clear"
        )
        adapter.trip_kill(reason)
        adapter._persisted_nav = float(persisted.get("nav", 0.0))
        report = adapter.seam_report()
        if not report["ok"]:
            raise OrderLedgerError(
                f"paper broker checkpoint failed seam verification: {report['errors']}"
            )
        return adapter, report


__all__ = [
    "LiveEndpointRefused",
    "PaperBrokerAdapter",
    "SeamReport",
    "refuse_live_endpoint",
]
