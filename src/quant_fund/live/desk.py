"""Live desk: pre-trade risk, order lifecycle, and a bridge.

Shadow is the default and transmits nothing. Loopback fills are SYNTHETIC and
stay in-process. Drop writes a ticket only when the venue is armed and
``DIP_LIVE_ARM=1``. Reports keep ``live_pnl_claim`` false and
``evidence_conditions`` blocked. Cash and positions are the risk book, not a
performance claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant_fund.formal.order_lifecycle import TERMINAL, Book, check_trace, step
from quant_fund.live.arming import (
    Arming,
    held_reason,
    read_arming,
    transmits,
    write_arming,
)
from quant_fund.live.bridge import BridgeError, BrokerEvent, DropBridge, LoopbackBridge
from quant_fund.live.exchange_clock import (
    ClockError,
    band_map,
    flag_map,
    move_entry,
    nyse_rth_closed,
    outside_band,
)
from quant_fund.live.fat_finger import fat_finger_reasons
from quant_fund.live.idempotency import assign_key
from quant_fund.live.market_data import (
    ActionPlan,
    MarketDataError,
    ReferenceSnapshot,
    assess_quote,
    jump_problem,
    plan_action,
    price_problem,
    require_internal,
    size_problems,
    snapshot_from_dict,
    snapshot_problem,
)
from quant_fund.live.market_data import (
    bind_symbol as bind_map,
)
from quant_fund.live.risk_limits import (
    install_ceilings,
    loosened_names,
    loss_reason,
    precision_reasons,
    structural_changes,
)
from quant_fund.pretrade.codes import (
    INTERNAL,
    KIND_CANCEL,
    KIND_ORDER,
    reason_names,
)
from quant_fund.pretrade.config import LimitConfig, PretradeConfig, sign_config
from quant_fund.pretrade.engine import OrderView, PretradeEngine

_EPS = 1e-8
_ZERO = "0" * 64
_SCHEMA = "live_desk_state.v1"
_SYMBOL_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
_CLIENT_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
_VISIBLE = {
    "new": "working",
    "partial": "partial",
    "filled": "filled",
    "canceled": "canceled",
    "rejected": "rejected",
    "expired": "canceled",
}


class LiveDeskError(ValueError):
    """The desk refused the call before a risk decision existed."""


@dataclass
class DeskOrder:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    qty: float
    px: float
    filled_qty: float = 0.0
    status: str = "working"
    lifecycle: str = "absent"
    venue: str = "shadow"
    synthetic: bool = False
    transmitted: bool = False
    acked: bool = False
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "px": self.px,
            "filled_qty": self.filled_qty,
            "status": self.status,
            "lifecycle": self.lifecycle,
            "venue": self.venue,
            "synthetic": self.synthetic,
            "transmitted": self.transmitted,
            "acked": self.acked,
            "reasons": list(self.reasons),
        }


def session_id(day: date) -> int:
    """Weekday index. Holidays are not removed; Saturday and Sunday share Friday.

    Same arithmetic as ``quant_fund.pretrade.shadow.business_session_id``.
    """
    zero = day.toordinal() - 1
    return (zero // 7) * 5 + min(zero % 7, 4) + 1


@dataclass
class LiveDesk:
    """One operator desk directory. Hold the lock until ``close``."""

    root: Path
    config: PretradeConfig
    hmac_key: bytes
    engine: PretradeEngine
    arming: Arming
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    mark_ts: dict[str, int] = field(default_factory=dict)
    reserved_qty: dict[str, float] = field(default_factory=dict)
    reserved_cash: float = 0.0
    orders: dict[str, DeskOrder] = field(default_factory=dict)
    order_ids: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    book: Book = field(default_factory=Book)
    seen: set[str] = field(default_factory=set)
    orphans: list[dict[str, str]] = field(default_factory=list)
    journal_hash: str = _ZERO
    halt_reason: str = ""
    trace_ok: bool = True
    peak: float = 0.0
    session_start: float = 0.0
    config_sha256: str = ""
    funded: bool = False
    rate_order_ts: list[int] = field(default_factory=list)
    rate_message_ts: list[int] = field(default_factory=list)
    dupes: list[tuple[str, str, int, int, int]] = field(default_factory=list)
    _loopback: LoopbackBridge = field(default_factory=LoopbackBridge)
    _drop: DropBridge | None = None
    _by_client: dict[str, str] = field(default_factory=dict)
    _lock_path: Path | None = None
    _closed: bool = False
    _integrity_frozen: bool = False
    _settled_before: float = 0.0
    controls: dict[str, tuple[bool, bool]] = field(default_factory=dict)
    _limit_override: LimitConfig | None = None
    symbol_map: dict[str, str] = field(default_factory=dict)
    feeds: dict[str, tuple[int | None, bool]] = field(default_factory=dict)
    reference: ReferenceSnapshot | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    action_halt: set[str] = field(default_factory=set)
    quote_ok: dict[str, bool] = field(default_factory=dict)
    bands: dict[str, tuple[float, float]] = field(default_factory=dict)
    _gate: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._drop = DropBridge(self.root)
        self._gate = threading.RLock()

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        hmac_key: bytes,
        config: PretradeConfig,
        cash: float | None = None,
        account_value: float | None = None,
        ts_ns: int | None = None,
    ) -> LiveDesk:
        """Open or create a desk. Locks the directory until ``close``."""
        desk_root = Path(root)
        desk_root.mkdir(parents=True, exist_ok=True)
        os.chmod(desk_root, 0o700)
        digest, _signature = sign_config(config, hmac_key)
        stamp = 0 if ts_ns is None else ts_ns
        if stamp < 0:
            raise LiveDeskError("ts_ns must be non-negative")
        # A missing cash grant is not buying power. The placeholder keeps the
        # engine constructible until the operator funds the desk.
        opening_cash = 1.0 if cash is None else _positive(cash, None)
        opening_account = 1.0 if account_value is None else _positive(account_value, None)
        engine = PretradeEngine(
            config,
            hmac_key=hmac_key,
            initial_nav=opening_account,
            initial_cash=opening_cash,
            ts_ns=stamp,
        )
        desk = cls(
            root=desk_root,
            config=config,
            hmac_key=hmac_key,
            engine=engine,
            arming=Arming("shadow"),
            cash=opening_cash,
            peak=opening_account,
            session_start=opening_account,
            config_sha256=digest,
        )
        desk._acquire()
        loaded = False
        try:
            desk.arming = read_arming(desk_root / "arming.json")
            desk._load(cash=cash, account_value=account_value, ts_ns=stamp)
            loaded = True
        finally:
            if not loaded:
                desk.close()
        return desk

    def close(self) -> None:
        self._closed = True
        path = self._lock_path
        self._lock_path = None
        if path is not None and path.exists():
            path.unlink()

    def __enter__(self) -> LiveDesk:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def status(self) -> dict[str, object]:
        self._alive()
        return self._envelope()

    def arm(self, venue: str, confirm: str, ts_ns: int) -> dict[str, object]:
        """Select a venue. Drop requires the confirm phrase. It still needs the env."""
        self._alive()
        self._clock(ts_ns)
        self._require_funded()
        if venue not in {"shadow", "loopback", "drop"}:
            raise LiveDeskError(f"unknown venue {venue!r}")
        if venue == "drop" and confirm != "SEND-LIVE-ORDERS":
            raise LiveDeskError("refusing drop venue without confirm SEND-LIVE-ORDERS")
        if venue != "shadow" and self.engine.book.killed:
            raise LiveDeskError("desk is halted; reset before arming a transmitting venue")
        self.arming = Arming(venue)
        write_arming(self.root / "arming.json", self.arming)
        self._journal("arm", {"venue": venue})
        self._save()
        return self._envelope(accepted=True, status="armed")

    def disarm(self, ts_ns: int) -> dict[str, object]:
        """Return to shadow. Orders already ticketed stay working until they fill or cancel."""
        self._alive()
        self._clock(ts_ns)
        self.arming = Arming("shadow")
        write_arming(self.root / "arming.json", self.arming)
        self._journal("disarm", {"venue": "shadow"})
        self._save()
        return self._envelope(accepted=True, status="shadow")

    def kill(self, actor: str, reason: str, ts_ns: int) -> dict[str, object]:
        with self._gate:
            self._alive()
            self._clock(ts_ns)
            self.engine.trip(actor, reason, ts_ns)
            self.halt_reason = reason
            canceled = self._cancel_working(ts_ns)
            still = self._working_ids()
            self._journal("kill", {"actor": actor, "reason": reason, "canceled": canceled})
            self._save()
            report = self._envelope(accepted=not still, status="halted")
            report["canceled"] = canceled
            report["still_working"] = still
            return report

    def set_symbol_controls(
        self,
        symbol: str,
        ts_ns: int,
        *,
        halted: bool | None = None,
        locate_ok: bool | None = None,
    ) -> dict[str, object]:
        """Halt a name or clear its locate. The next order sees it without a restart."""
        with self._gate:
            self._alive()
            self._clock(ts_ns)
            symbol = _symbol(symbol)
            halt, locate = self._symbol_flags(symbol)
            if halted is not None:
                halt = bool(halted)
            if locate_ok is not None:
                locate = bool(locate_ok)
            self.controls[symbol] = (halt, locate)
            if symbol in self.marks:
                self._project(ts_ns)
            self._journal(
                "symbol_controls", {"symbol": symbol, "halted": halt, "locate_ok": locate}
            )
            self._save()
            return self._envelope(accepted=True, status="controls", halted=halt, locate_ok=locate)

    def update_limits(
        self, limits: LimitConfig, ts_ns: int, *, reauthorize: bool = False
    ) -> dict[str, object]:
        """Apply ceilings now. A looser ceiling needs ``reauthorize``."""
        with self._gate:
            self._alive()
            self._require_funded()
            self._clock(ts_ns)
            structural = structural_changes(self.config.limits, limits)
            if structural:
                return self._limit_envelope(applied=False, structural=structural)
            looser = loosened_names(self.config.limits, limits)
            if looser and not reauthorize:
                return self._limit_envelope(applied=False, loosened=looser)
            self._apply_limits(limits)
            self._latch_loss(ts_ns)
            self._journal("limits", {"reauthorized": reauthorize, "loosened": list(looser)})
            self._save()
            return self._limit_envelope(applied=True, loosened=looser, reauthorized=reauthorize)

    def reset(self, actor: str, reason: str, ts_ns: int) -> dict[str, object]:
        """Clear a halt. An integrity failure or an internal fault stays halted."""
        self._alive()
        self._clock(ts_ns)
        if self._integrity_frozen:
            raise LiveDeskError(self.halt_reason or "journal integrity failure")
        if int(self.engine.book.kill_reason) & INTERNAL:
            raise LiveDeskError("internal fault; start a new desk directory")
        self.engine.reset(actor, reason, ts_ns)
        self.halt_reason = ""
        self._journal("reset", {"actor": actor, "reason": reason})
        self._save()
        report = self._envelope(accepted=True, status="reset")
        report["transmission_after_reset"] = self._transmits()
        return report

    def mark(self, symbol: str, px: float, ts_ns: int) -> dict[str, object]:
        with self._gate:
            self._alive()
            self._clock(ts_ns)
            symbol = _symbol(symbol)
            _price(px)
            if self._quote_is_older(symbol, ts_ns):
                raise LiveDeskError("mark timestamp is older than the stored quote")
            problem = jump_problem(self.marks.get(symbol), px)
            if problem:
                raise LiveDeskError(problem)
            self._bind_identity(symbol)
            self.marks[symbol] = px
            self.mark_ts[symbol] = ts_ns
            self._project(ts_ns)
            self._latch_loss(ts_ns)
            self._collect_matured()
            self._journal("mark", {"symbol": symbol, "px": px, "ts_ns": ts_ns})
            self._save()
            return self._envelope(accepted=True, status="marked")

    def bind_symbol(self, internal: str, broker: str, ts_ns: int) -> dict[str, object]:
        """Bind one internal id to one broker symbol. A second binding is an error."""
        with self._gate:
            self._alive()
            self._clock(ts_ns)
            left = _symbol(internal)
            try:
                bind_map(self.symbol_map, left, broker)
            except MarketDataError as exc:
                raise LiveDeskError(str(exc)) from exc
            self._journal("symbol_map", {"internal": left, "broker": self.symbol_map[left]})
            self._save()
            return self._envelope(accepted=True, status="symbol_map", symbol=left)

    def ingest_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        seq: int,
        ts_ns: int,
        *,
        snapshot: bool = False,
    ) -> dict[str, object]:
        """Accept a sequenced quote or leave the stored mark untouched."""
        with self._gate:
            self._alive()
            self._clock(ts_ns)
            return self._ingest_locked(symbol, bid, ask, seq, ts_ns, snapshot=snapshot)

    def set_reference(self, snapshot: ReferenceSnapshot, ts_ns: int) -> dict[str, object]:
        """Install the lot and tick snapshot used on the next order."""
        with self._gate:
            self._alive()
            self._clock(ts_ns)
            self._require_reference(snapshot)
            self.reference = snapshot
            self._journal("reference", {"version": snapshot.version})
            self._save()
            return self._envelope(accepted=True, status="reference", version=snapshot.version)

    def set_price_band(
        self, symbol: str, lower: float, upper: float, ts_ns: int
    ) -> dict[str, object]:
        """Install an inclusive LULD-style band. Prices outside it are not tradable."""
        with self._gate:
            self._alive()
            self._clock(ts_ns)
            cleaned = _symbol(symbol)
            self.bands[cleaned] = _valid_band(lower, upper)
            self._journal("price_band", {"symbol": cleaned, "lower": lower, "upper": upper})
            self._save()
            return self._envelope(
                accepted=True, status="price_band", symbol=cleaned, lower=lower, upper=upper
            )

    def apply_corporate_action(
        self,
        kind: str,
        symbol: str,
        ts_ns: int,
        *,
        ratio: float | None = None,
        cash_per_share: float | None = None,
        new_symbol: str | None = None,
    ) -> dict[str, object]:
        """Adjust one name, or halt it, without changing account value."""
        with self._gate:
            self._alive()
            self._require_funded()
            self._clock(ts_ns)
            return self._apply_action_locked(
                kind,
                symbol,
                ts_ns,
                ratio=ratio,
                cash_per_share=cash_per_share,
                new_symbol=new_symbol,
            )

    def submit(
        self,
        symbol: str,
        side: str,
        qty: float,
        px: float,
        ts_ns: int,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, object]:
        """Risk-check one limit order. Transmit only when the venue switch is on."""
        with self._gate:
            return self._submit_body(symbol, side, qty, px, ts_ns, client_order_id=client_order_id)

    def _submit_body(
        self,
        symbol: str,
        side: str,
        qty: float,
        px: float,
        ts_ns: int,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, object]:
        self._alive()
        self._require_funded()
        self._clock(ts_ns)
        symbol = _symbol(symbol)
        side = _side(side)
        _quantity(qty)
        _price(px)
        client = assign_key(
            symbol=symbol,
            side=side,
            quantity=qty,
            limit_price=px,
            client_order_id=client_order_id,
            known=tuple((item.client_order_id, item.status) for item in self.orders.values()),
        )
        _client(client)
        prior = self._by_client.get(client)
        if prior is not None:
            order = self.orders[prior]
            return self._envelope(
                accepted=order.status not in {"denied", "rejected"},
                status=order.status,
                transmitted=order.transmitted,
                idempotent=True,
                order=order.as_dict(),
            )
        blocked = self._market_block(symbol, ts_ns, qty, px)
        if blocked:
            return self._deny(symbol, side, qty, px, ts_ns, client, blocked)
        prior_mark = self.marks.get(symbol)
        finger = fat_finger_reasons(
            quantity=qty,
            limit_price=px,
            mark=prior_mark if prior_mark is not None else px,
            max_quantity=float(self.config.limits.max_order_quantity),
            max_notional=float(self.config.limits.max_order_notional),
            collar_bps=float(self.config.limits.price_collar_bps),
        )
        self._bind_identity(symbol)
        self.marks[symbol] = px
        self.mark_ts[symbol] = ts_ns
        self._project(ts_ns)
        view = self._view(symbol, side, qty, px, ts_ns, KIND_ORDER)
        decision = self.engine.decide(view, apply=False)
        self._collect_matured()
        reasons = list(decision.reasons)
        reasons.extend(
            precision_reasons(
                qty,
                px,
                float(self.config.limits.qty_tick),
                float(self.config.limits.price_tick),
            )
        )
        if self._rate_blocked(ts_ns, order=True) and "order_rate" not in reasons:
            reasons.append("order_rate")
        if self._duplicate(symbol, side, qty, px, ts_ns) and "duplicate_order" not in reasons:
            reasons.append("duplicate_order")
        _merge_reasons(reasons, finger)
        self._note_rate(ts_ns, order=True)
        self._note_dupe(symbol, side, qty, px, ts_ns)
        order = DeskOrder(
            order_id=uuid.uuid4().hex,
            client_order_id=client,
            symbol=symbol,
            side=side,
            qty=qty,
            px=px,
            venue=self.arming.venue,
            synthetic=self.arming.venue == "loopback",
            reasons=tuple(reasons),
        )
        self._store(order)
        if reasons or not decision.allowed:
            self._drop_unaccepted_mark(symbol, prior_mark)
            order.status = "denied"
            self._journal("deny", order.as_dict())
            self._save()
            return self._envelope(
                accepted=False,
                status="denied",
                transmitted=False,
                order=order.as_dict(),
            )
        if not self._transmits():
            order.status = "held" if self.arming.venue == "drop" else "shadowed"
            self._journal(order.status, order.as_dict())
            self._save()
            return self._envelope(
                accepted=True,
                status=order.status,
                transmitted=False,
                held_reason=held_reason(self.arming, os.environ),
                order=order.as_dict(),
            )
        self._reserve(symbol, side, qty, px)
        self._step({"op": "submit", "order_id": order.order_id, "qty": qty})
        self._sync_order(order)
        order.transmitted = True
        self._send(_ticket(order))
        self._journal("transmit", order.as_dict())
        self._save()
        if self.arming.venue == "loopback":
            self.poll(ts_ns)
            order = self.orders[order.order_id]
        return self._envelope(
            accepted=order.status not in {"denied", "rejected"},
            status=order.status,
            transmitted=True,
            order=order.as_dict(),
        )

    def cancel(self, order_id: str, ts_ns: int) -> dict[str, object]:
        self._alive()
        self._clock(ts_ns)
        order = self.orders.get(order_id)
        if order is None or order.lifecycle not in {"new", "partial"}:
            return self._envelope(accepted=False, status="unknown", transmitted=False)
        if self._going_backwards(ts_ns):
            return self._envelope(
                accepted=False, status=order.status, reasons=["non_monotonic_time"]
            )
        if order.transmitted and not self._transmits():
            return self._envelope(
                accepted=False,
                status=order.status,
                transmitted=False,
                held_reason=held_reason(self.arming, os.environ),
                order=order.as_dict(),
            )
        self._project(ts_ns)
        view = self._view(order.symbol, order.side, order.qty, order.px, ts_ns, KIND_CANCEL)
        decision = self.engine.decide(view, apply=False)
        self._collect_matured()
        reasons = list(decision.reasons)
        if self._rate_blocked(ts_ns, order=False) and "message_rate" not in reasons:
            reasons.append("message_rate")
        self._note_rate(ts_ns, order=False)
        if reasons or not decision.allowed:
            self._save()
            return self._envelope(
                accepted=False,
                status=order.status,
                reasons=reasons,
                order=order.as_dict(),
            )
        self._send(_cancel_ticket(order))
        self._journal("cancel_transmit", {"order_id": order.order_id})
        self._save()
        if self.arming.venue == "loopback":
            self.poll(ts_ns)
            order = self.orders[order_id]
        return self._envelope(accepted=True, status=order.status, order=order.as_dict())

    def poll(self, ts_ns: int) -> dict[str, object]:
        """Apply bridge events. Fills are applied even while halted; new orders are not."""
        self._alive()
        applied: list[dict[str, object]] = []
        try:
            events = self._events()
        except BridgeError as exc:
            self._freeze(str(exc))
            return self._envelope(accepted=False, error=str(exc), applied=applied)
        for event in events:
            if event.event_id in self.seen:
                continue
            applied.append(self._apply_event(event, ts_ns))
            if self._integrity_frozen:
                break
            self._save()
        return self._envelope(accepted=True, applied=applied)

    def flatten(self, ts_ns: int) -> dict[str, object]:
        """Submit reduce-only limits at the last mark. Each one still passes pre-trade."""
        self._alive()
        self._require_funded()
        reports: list[object] = []
        for symbol, qty in list(self.positions.items()):
            if abs(qty) <= _EPS:
                continue
            mark = self.marks.get(symbol)
            if mark is None or mark <= 0.0:
                reports.append({"symbol": symbol, "accepted": False, "reasons": ["stale_data"]})
                continue
            side = "SELL" if qty > 0.0 else "BUY"
            reports.append(self.submit(symbol, side, abs(qty), mark, ts_ns))
            ts_ns += 1
        return self._envelope(flatten=reports)

    def _apply_event(self, event: BrokerEvent, ts_ns: int) -> dict[str, object]:
        self.seen.add(event.event_id)
        order = self.orders.get(event.order_id)
        if order is None:
            self.orphans.append({"order_id": event.order_id, "event_id": event.event_id})
            self._latch(f"orphan {event.kind} {event.event_id}")
            self._journal("orphan", {"event_id": event.event_id, "order_id": event.order_id})
            return {"event_id": event.event_id, "result": "orphan"}
        if event.kind == "ack":
            order.acked = True
            self._journal("ack", {"order_id": order.order_id, "event_id": event.event_id})
            return {"event_id": event.event_id, "result": "ack"}
        if order.lifecycle in TERMINAL or order.lifecycle == "absent":
            self._latch(f"{event.kind} on {order.lifecycle} order {order.order_id}")
            self._journal("unapplied", {"event_id": event.event_id, "order_id": order.order_id})
            return {"event_id": event.event_id, "result": "unapplied"}
        if event.kind == "reject":
            self._release_remaining(order)
            self._step({"op": "reject", "order_id": order.order_id})
            self._sync_order(order)
            self._journal("reject", {"order_id": order.order_id, "reason": event.reason})
            return {"event_id": event.event_id, "result": "reject"}
        if event.kind == "cancel":
            self._release_remaining(order)
            self._step({"op": "cancel", "order_id": order.order_id})
            self._sync_order(order)
            self._journal("cancel", {"order_id": order.order_id})
            return {"event_id": event.event_id, "result": "cancel"}
        remaining = order.qty - order.filled_qty
        if event.qty > remaining + _EPS:
            self._latch(f"fill {event.event_id} exceeds remaining quantity")
            self._journal("unapplied", {"event_id": event.event_id, "order_id": order.order_id})
            return {"event_id": event.event_id, "result": "unapplied"}
        rest = order.filled_qty + event.qty < order.qty - _EPS
        self._release(order.symbol, order.side, event.qty, order.px)
        pos_before = self.positions.get(order.symbol, 0.0)
        signed = event.qty if order.side == "BUY" else -event.qty
        self.positions[order.symbol] = pos_before + signed
        self._note_fill(order, event, ts_ns, pos_before)
        self._step(
            {
                "op": "fill",
                "order_id": order.order_id,
                "fill_id": event.event_id,
                "qty": event.qty,
                "rest": rest,
            }
        )
        self._sync_order(order)
        self._note_loss(ts_ns)
        if event.synthetic:
            order.synthetic = True
        self._check_books()
        self._journal(
            "fill",
            {
                "order_id": order.order_id,
                "event_id": event.event_id,
                "qty": event.qty,
                "price": event.price,
                "synthetic": event.synthetic,
            },
        )
        return {"event_id": event.event_id, "result": "fill", "synthetic": event.synthetic}

    def _note_fill(
        self, order: DeskOrder, event: BrokerEvent, ts_ns: int, pos_before: float
    ) -> None:
        available = self.cash
        self.engine.book.settled = available
        side = 1 if order.side == "BUY" else -1
        self.engine.note_fill(
            symbol_id=self._sid(order.symbol),
            side=side,
            qty=event.qty,
            px=event.price,
            fee=event.fee,
            ts_ns=ts_ns,
            session_id=self._session(ts_ns),
            pos_before=pos_before,
        )
        self.cash = float(self.engine.book.settled)
        if self.engine.book.killed and not self.halt_reason:
            self.halt_reason = "pretrade engine halted while applying a fill"
        cost = event.qty * event.price + event.fee
        if order.side == "BUY" and cost > available + _EPS:
            self._latch("fill price exceeded settled cash")

    def _send(self, ticket: dict[str, object]) -> None:
        try:
            if self.arming.venue == "loopback":
                self._loopback.transmit(ticket)
            elif self.arming.venue == "drop" and self._drop is not None:
                self._drop.transmit(ticket)
            else:
                raise LiveDeskError("no bridge for the armed venue")
        except (BridgeError, OSError, LiveDeskError) as exc:
            self._latch(f"transmit failed: {exc}")
            raise LiveDeskError(str(exc)) from exc

    def _events(self) -> list[BrokerEvent]:
        events = list(self._loopback.poll())
        if self._drop is not None:
            events.extend(self._drop.poll())
        return events

    def _transmits(self) -> bool:
        return transmits(
            self.arming,
            os.environ,
            halted=bool(self.engine.book.killed) or self._integrity_frozen,
            trace_ok=self.trace_ok,
        )

    def _symbol_flags(self, symbol: str) -> tuple[bool, bool]:
        return self.controls.get(symbol, (False, True))

    def _latch_loss(self, ts_ns: int) -> None:
        if self.engine.book.killed:
            return
        nav = float(self.engine.book.nav)
        if nav != nav or not nav > 0.0:
            return
        reason = loss_reason(
            nav, float(self.engine.book.daily_floor), float(self.engine.book.dd_floor)
        )
        if not reason:
            return
        self.engine.trip("risk", reason, ts_ns)
        self.halt_reason = reason
        self._cancel_working(ts_ns)

    def _note_loss(self, ts_ns: int) -> None:
        nav = self._account_value()
        if nav == nav and nav > 0.0:
            self.engine.book.nav = nav
            if nav > float(self.engine.book.peak):
                self.engine.book.peak = nav
                self.peak = nav
                self.engine._recompute_floors()
        self._latch_loss(ts_ns)

    def _cancel_working(self, ts_ns: int) -> list[str]:
        """Cancel every transmitted working order. Only confirmed cancels count."""
        targets = [
            order
            for order in list(self.orders.values())
            if order.lifecycle in {"new", "partial"} and order.transmitted
        ]
        for order in targets:
            self._send(_cancel_ticket(order))
        if targets and self.arming.venue == "loopback":
            self.poll(ts_ns)
        return [
            order.order_id
            for order in targets
            if order.lifecycle not in {"new", "partial"} or not order.transmitted
        ]

    def _working_ids(self) -> list[str]:
        return [
            order.order_id
            for order in self.orders.values()
            if order.lifecycle in {"new", "partial"} and order.transmitted
        ]

    def _apply_limits(self, limits: LimitConfig) -> None:
        install_ceilings(self.engine, limits)
        self.config = self.config.model_copy(update={"limits": limits})
        digest, signature = sign_config(self.config, self.hmac_key)
        self.engine.config_sha256 = digest
        self.engine.config_hmac_sha256 = signature
        self._limit_override = limits

    def _limit_envelope(
        self,
        *,
        applied: bool,
        loosened: tuple[str, ...] = (),
        structural: tuple[str, ...] = (),
        reauthorized: bool = False,
    ) -> dict[str, object]:
        status = "limits" if applied else "limits_refused"
        return self._envelope(
            accepted=applied,
            status=status,
            applied=applied,
            loosened=list(loosened),
            structural=list(structural),
            reauthorized=reauthorized,
        )

    def _restore_market(self, raw: Mapping[str, Any]) -> None:
        self.actions = _action_list(raw.get("corporate_actions"))
        halted = raw.get("action_halt") or []
        if not isinstance(halted, list):
            raise LiveDeskError("action halt set is malformed")
        self.action_halt = {str(item) for item in halted}
        self.feeds = _feed_map(raw.get("feeds"))
        self._load_clock(raw)
        self.reference = _reference_from_state(raw.get("reference"))
        if "symbol_map" not in raw:
            self.symbol_map = {}
            self._seed_identity()
            return
        payload = raw.get("symbol_map")
        if not isinstance(payload, dict):
            raise LiveDeskError("symbol map is malformed")
        self.symbol_map = {str(key): str(value) for key, value in payload.items()}

    def _load_clock(self, raw: Mapping[str, Any]) -> None:
        try:
            self.quote_ok = flag_map(raw.get("quote_ok"))
            self.bands = band_map(raw.get("price_bands"))
        except ClockError as exc:
            raise LiveDeskError(str(exc)) from exc

    def _seed_identity(self) -> None:
        names = set(self.positions) | set(self.marks)
        names.update(order.symbol for order in self.orders.values())
        for name in sorted(names):
            if name not in self.symbol_map:
                bind_map(self.symbol_map, name, name)

    def _restore_controls(self, raw: Mapping[str, Any]) -> None:
        payload = raw.get("symbol_controls") or {}
        if not isinstance(payload, dict):
            raise LiveDeskError("symbol controls are malformed")
        for symbol, item in payload.items():
            if not isinstance(item, dict):
                raise LiveDeskError("symbol control is malformed")
            self.controls[str(symbol)] = (
                bool(item.get("halted")),
                bool(item.get("locate_ok", True)),
            )

    def _restore_saved_limits(self, raw: Mapping[str, Any]) -> None:
        payload = raw.get("limit_override")
        if payload is None:
            return
        if not isinstance(payload, dict):
            raise LiveDeskError("stored limit override is malformed")
        self._apply_limits(LimitConfig.model_validate(payload))

    def _project(self, ts_ns: int) -> None:
        self.engine.refresh_session(ts_ns)
        symbols = set(self.positions) | set(self.reserved_qty) | set(self.marks)
        for symbol in sorted(symbols):
            sid = self.engine.ensure_symbol(symbol)
            if sid < 0:
                raise LiveDeskError("symbol table is full")
            pos = self.positions.get(symbol, 0.0) + self.reserved_qty.get(symbol, 0.0)
            ref = self.marks.get(symbol, float("nan"))
            halted, locate_ok = self._symbol_flags(symbol)
            self.engine.set_symbol(
                sid,
                pos=pos,
                ref_px=ref,
                ref_ts_ns=self.mark_ts.get(symbol, 0),
                halted=halted,
                locate_ok=locate_ok,
            )
        account = self._account_value()
        self.engine.update_account(nav=account, mark_ts_ns=ts_ns if account > 0.0 else 0)
        self.peak = float(self.engine.book.peak)
        self.session_start = float(self.engine.book.session_start)
        self._settled_before = self.cash - self.reserved_cash
        self.engine.book.settled = self._settled_before

    def _collect_matured(self) -> None:
        matured = float(self.engine.book.settled) - self._settled_before
        if matured > _EPS:
            self.cash += matured

    def _view(
        self, symbol: str, side: str, qty: float, px: float, ts_ns: int, kind: int
    ) -> OrderView:
        return OrderView(
            symbol_id=self._sid(symbol),
            side=1 if side == "BUY" else -1,
            qty=qty,
            px=px,
            ts_ns=ts_ns,
            session_id=self._session(ts_ns),
            is_limit=1,
            kind=kind,
        )

    def _sid(self, symbol: str) -> int:
        sid = self.engine.ensure_symbol(symbol)
        if sid < 0:
            raise LiveDeskError("symbol table is full")
        return sid

    def _session(self, ts_ns: int) -> int:
        local = datetime.fromtimestamp(
            ts_ns / 1_000_000_000, ZoneInfo(self.config.session.timezone)
        )
        return session_id(local.date())

    def _reserve(self, symbol: str, side: str, qty: float, px: float) -> None:
        signed = qty if side == "BUY" else -qty
        self.reserved_qty[symbol] = self.reserved_qty.get(symbol, 0.0) + signed
        if side == "BUY":
            self.reserved_cash += qty * px

    def _release(self, symbol: str, side: str, qty: float, px: float) -> None:
        if qty <= _EPS:
            return
        signed = qty if side == "BUY" else -qty
        left = self.reserved_qty.get(symbol, 0.0) - signed
        if abs(left) <= _EPS:
            self.reserved_qty.pop(symbol, None)
        else:
            self.reserved_qty[symbol] = left
        if side == "BUY":
            self.reserved_cash = max(0.0, self.reserved_cash - qty * px)
            if self.reserved_cash <= _EPS:
                self.reserved_cash = 0.0

    def _release_remaining(self, order: DeskOrder) -> None:
        self._release(order.symbol, order.side, order.qty - order.filled_qty, order.px)

    def _step(self, event: dict[str, Any]) -> None:
        try:
            self.book = step(self.book, event)
        except Exception as exc:
            self.trace_ok = False
            self._latch(f"lifecycle refused {event.get('op')}: {exc}")
            raise LiveDeskError(str(exc)) from exc
        self.trace.append(event)
        checked = check_trace(self.trace)
        self.trace_ok = checked.ok
        if not checked.ok:
            self._latch("lifecycle trace failed: " + "; ".join(checked.violations))
            raise LiveDeskError("lifecycle trace failed")

    def _sync_order(self, order: DeskOrder) -> None:
        state = self.book.orders.get(order.order_id)
        if state is None:
            return
        order.lifecycle = state.status
        order.filled_qty = float(state.filled)
        order.status = _VISIBLE.get(state.status, state.status)

    def _store(self, order: DeskOrder) -> None:
        self.orders[order.order_id] = order
        self.order_ids.append(order.order_id)
        self._by_client[order.client_order_id] = order.order_id

    def _deny(
        self,
        symbol: str,
        side: str,
        qty: float,
        px: float,
        ts_ns: int,
        client: str,
        reasons: tuple[str, ...],
    ) -> dict[str, object]:
        del ts_ns
        order = DeskOrder(
            order_id=uuid.uuid4().hex,
            client_order_id=client,
            symbol=symbol,
            side=side,
            qty=qty,
            px=px,
            status="denied",
            venue=self.arming.venue,
            reasons=reasons,
        )
        self._store(order)
        self._journal("deny", order.as_dict())
        self._save()
        return self._envelope(
            accepted=False, status="denied", transmitted=False, order=order.as_dict()
        )

    def _check_books(self) -> None:
        expected: dict[str, float] = {}
        for order in self.orders.values():
            if order.filled_qty <= _EPS:
                continue
            signed = order.filled_qty if order.side == "BUY" else -order.filled_qty
            expected[order.symbol] = expected.get(order.symbol, 0.0) + signed
        symbols = set(expected) | {k for k, v in self.positions.items() if abs(v) > _EPS}
        for symbol in symbols:
            if abs(self.positions.get(symbol, 0.0) - expected.get(symbol, 0.0)) > _EPS:
                self._latch("position book diverged from fills")
                raise LiveDeskError("position book diverged from fills")
        reserved, cash = self._expected_reserved()
        for symbol in set(reserved) | set(self.reserved_qty):
            if abs(self.reserved_qty.get(symbol, 0.0) - reserved.get(symbol, 0.0)) > _EPS:
                self._latch("reservation diverged from working orders")
                raise LiveDeskError("reservation diverged from working orders")
        if abs(self.reserved_cash - cash) > 1e-6:
            self._latch("reserved cash diverged from working buys")
            raise LiveDeskError("reserved cash diverged from working buys")

    def _expected_reserved(self) -> tuple[dict[str, float], float]:
        qty: dict[str, float] = {}
        cash = 0.0
        for order in self.orders.values():
            if order.lifecycle not in {"new", "partial"}:
                continue
            remaining = order.qty - order.filled_qty
            if remaining <= _EPS:
                continue
            signed = remaining if order.side == "BUY" else -remaining
            qty[order.symbol] = qty.get(order.symbol, 0.0) + signed
            if order.side == "BUY":
                cash += remaining * order.px
        return qty, cash

    def _duplicate(self, symbol: str, side: str, qty: float, px: float, ts_ns: int) -> bool:
        window = int(self.config.limits.duplicate_window_ns)
        fp = _ticks(self.config, symbol, side, qty, px)
        return any(
            (row[0], row[1], row[2], row[3]) == fp and ts_ns - row[4] <= window
            for row in self.dupes
        )

    def _note_dupe(self, symbol: str, side: str, qty: float, px: float, ts_ns: int) -> None:
        fp = _ticks(self.config, symbol, side, qty, px)
        self.dupes.append((fp[0], fp[1], fp[2], fp[3], ts_ns))
        window = int(self.config.limits.duplicate_window_ns)
        self.dupes = [row for row in self.dupes if ts_ns - row[4] <= window]

    def _note_rate(self, ts_ns: int, *, order: bool) -> None:
        self.rate_message_ts.append(ts_ns)
        if order:
            self.rate_order_ts.append(ts_ns)
        cap = int(self.config.limits.ring_capacity)
        self.rate_message_ts = self.rate_message_ts[-cap:]
        self.rate_order_ts = self.rate_order_ts[-cap:]

    def _rate_blocked(self, ts_ns: int, *, order: bool) -> bool:
        window = int(self.config.limits.rate_window_ns)
        if order:
            count = sum(1 for stamp in self.rate_order_ts if ts_ns - stamp <= window)
            if count >= int(self.config.limits.max_orders_per_window):
                return True
        count = sum(1 for stamp in self.rate_message_ts if ts_ns - stamp <= window)
        return count >= int(self.config.limits.max_messages_per_window)

    def _ingest_locked(
        self,
        symbol: str,
        bid: float,
        ask: float,
        seq: int,
        ts_ns: int,
        *,
        snapshot: bool,
    ) -> dict[str, object]:
        try:
            cleaned = require_internal(self.symbol_map, symbol)
        except MarketDataError as exc:
            raise LiveDeskError(str(exc)) from exc
        if self._nyse_closed(ts_ns):
            self._journal(
                "quote", {"symbol": cleaned, "accepted": False, "reasons": ["market_closed"]}
            )
            self._save()
            return self._envelope(
                accepted=False,
                status="quote",
                symbol=cleaned,
                stale=False,
                reasons=["market_closed"],
                mark=self.marks.get(cleaned),
            )
        previous_seq, was_stale = self.feeds.get(cleaned, (None, False))
        decision = assess_quote(
            bid=float(bid),
            ask=float(ask),
            seq=int(seq),
            previous_seq=previous_seq,
            previous_mark=self.marks.get(cleaned),
            was_stale=was_stale,
            snapshot=snapshot,
        )
        accepted, stale, reasons = self._store_quote(cleaned, decision, ts_ns)
        self._journal(
            "quote",
            {"symbol": cleaned, "accepted": accepted, "reasons": reasons},
        )
        self._save()
        return self._envelope(
            accepted=accepted,
            status="quote",
            symbol=cleaned,
            stale=stale,
            reasons=reasons,
            mark=self.marks.get(cleaned),
        )

    def _apply_action_locked(
        self,
        kind: str,
        symbol: str,
        ts_ns: int,
        *,
        ratio: float | None,
        cash_per_share: float | None,
        new_symbol: str | None,
    ) -> dict[str, object]:
        cleaned = _symbol(symbol)
        plan = plan_action(
            kind=kind,
            symbol=cleaned,
            cash=float(self.cash),
            positions=dict(self.positions),
            marks=dict(self.marks),
            working=self._symbol_working(cleaned),
            ratio=ratio,
            cash_per_share=cash_per_share,
            new_symbol=new_symbol,
            qty_cap=float(self.config.limits.max_position_quantity),
        )
        if plan.apply:
            self._commit_action(plan, ts_ns)
            self._project(ts_ns)
            self._latch_loss(ts_ns)
        elif plan.halt:
            self._halt_for_action(plan.symbol, ts_ns)
        record = _action_record(kind, plan, ts_ns)
        self.actions.append(record)
        self._journal("corporate_action", record)
        self._save()
        return self._envelope(
            accepted=plan.apply,
            status="corporate_action",
            reason=plan.reason,
            symbol=plan.symbol,
            new_symbol=plan.new_symbol,
            halted=bool(self.controls.get(plan.new_symbol, (False, True))[0]),
        )

    def _commit_action(self, plan: ActionPlan, ts_ns: int) -> None:
        self.cash = float(plan.cash)
        self.positions = {str(key): float(value) for key, value in plan.positions.items()}
        self.marks = {str(key): float(value) for key, value in plan.marks.items()}
        self._scale_desk_orders(plan)
        if plan.new_symbol != plan.symbol:
            self._rename_desk(plan.symbol, plan.new_symbol, plan.qty_factor, ts_ns)
        else:
            self._scale_lots(plan.symbol, plan.qty_factor)
        self._clear_action_halt(plan, ts_ns)

    def _halt_for_action(self, symbol: str, ts_ns: int) -> None:
        self.action_halt.add(symbol)
        halt, locate = self._symbol_flags(symbol)
        del halt
        self.controls[symbol] = (True, locate)
        if symbol in self.marks:
            self._project(ts_ns)

    def _clear_action_halt(self, plan: ActionPlan, ts_ns: int) -> None:
        if plan.symbol not in self.action_halt and plan.new_symbol not in self.action_halt:
            return
        self.action_halt.discard(plan.symbol)
        self.action_halt.discard(plan.new_symbol)
        _halt, locate = self._symbol_flags(plan.new_symbol)
        self.controls[plan.new_symbol] = (False, locate)

    def _scale_desk_orders(self, plan: ActionPlan) -> None:
        factor = float(plan.qty_factor)
        for order in self.orders.values():
            if order.symbol != plan.symbol:
                continue
            if factor != 1.0:
                order.qty *= factor
                order.filled_qty *= factor
                order.px /= factor
            if plan.new_symbol != plan.symbol:
                order.symbol = plan.new_symbol
        if factor != 1.0 and plan.symbol in self.reserved_qty:
            self.reserved_qty[plan.symbol] = float(self.reserved_qty[plan.symbol]) * factor
        if plan.new_symbol == plan.symbol:
            return
        if plan.symbol in self.reserved_qty:
            self.reserved_qty[plan.new_symbol] = self.reserved_qty.pop(plan.symbol)
        if plan.symbol in self.mark_ts:
            self.mark_ts[plan.new_symbol] = self.mark_ts.pop(plan.symbol)

    def _rename_desk(self, old: str, new: str, factor: float, ts_ns: int) -> None:
        broker_name = self.symbol_map.pop(old, old)
        self.symbol_map[new] = new if broker_name == old else broker_name
        feed = self.feeds.pop(old, None)
        if feed is not None:
            self.feeds[new] = feed
        if old in self.controls:
            self.controls[new] = self.controls.pop(old)
        move_entry(self.quote_ok, old, new)
        move_entry(self.bands, old, new)
        self._move_lots(old, new, factor, ts_ns)

    def _move_lots(self, old: str, new: str, factor: float, ts_ns: int) -> None:
        sid = self.engine._ids.get(old)
        if sid is None:
            return
        sym = self.engine.book.syms[sid]
        lots = [
            (float(sym.lq[index]) * factor, int(sym.ls[index]), int(sym.lu[index]))
            for index in range(int(sym.nlot))
        ]
        self.engine.set_symbol(sid, pos=0.0, ref_px=1.0, ref_ts_ns=ts_ns, locate_ok=True)
        for index in range(int(sym.nlot)):
            sym.lq[index] = 0.0
        sym.nlot = 0
        new_sid = self._sid(new)
        for qty, settles, funded in lots:
            if qty > 0.0:
                self.engine.restore_lot(
                    new_sid, qty=qty, settles_ns=settles, unsettled_funded=bool(funded)
                )

    def _scale_lots(self, symbol: str, factor: float) -> None:
        if factor == 1.0:
            return
        sid = self.engine._ids.get(symbol)
        if sid is None:
            return
        sym = self.engine.book.syms[sid]
        for index in range(int(sym.nlot)):
            sym.lq[index] = float(sym.lq[index]) * factor

    def _market_block(self, symbol: str, ts_ns: int, qty: float, px: float) -> tuple[str, ...]:
        if self._going_backwards(ts_ns):
            return ("non_monotonic_time",)
        if self._nyse_closed(ts_ns):
            return ("market_closed",)
        if self._quote_blocks(symbol):
            return ("invalid_quote",)
        if self._mark_is_stale(symbol, ts_ns) or self._feed_stale(symbol):
            return ("stale_data",)
        reasons: list[str] = []
        jumped = jump_problem(self.marks.get(symbol), px)
        if jumped:
            reasons.append(jumped)
        if outside_band(px, self.bands.get(symbol)):
            reasons.append("luld")
        ref = snapshot_problem(self.reference, symbol, ts_ns)
        if ref:
            reasons.append(ref)
        elif self.reference is not None:
            reasons.extend(
                size_problems(
                    qty, px, self.reference.lot_size[symbol], self.reference.tick_size[symbol]
                )
            )
        return tuple(reasons)

    def _feed_stale(self, symbol: str) -> bool:
        row = self.feeds.get(symbol)
        return row is not None and bool(row[1])

    def _nyse_closed(self, ts_ns: int) -> bool:
        clock = self.config.session
        if clock.timezone != "America/New_York" or clock.weekends_open:
            return False
        return nyse_rth_closed(ts_ns)

    def _quote_blocks(self, symbol: str) -> bool:
        """A rejected quote with no stored mark must not authorize the next order."""
        return self.quote_ok.get(symbol) is False and symbol not in self.marks

    def _store_quote(self, symbol: str, decision: Any, ts_ns: int) -> tuple[bool, bool, list[str]]:
        had_mark = symbol in self.marks
        mark = decision.mark
        accept = bool(decision.accept and mark is not None)
        reasons = list(decision.reasons)
        if accept and outside_band(float(mark), self.bands.get(symbol)):
            accept = False
            reasons.append("luld")
        self.feeds[symbol] = (decision.seq, decision.stale)
        if accept and mark is not None:
            price = float(mark)
            self.quote_ok[symbol] = True
            self.marks[symbol] = price
            self.mark_ts[symbol] = ts_ns
            self._project(ts_ns)
            self._latch_loss(ts_ns)
        elif not had_mark:
            self.quote_ok[symbol] = False
        return accept, bool(decision.stale), reasons

    def _symbol_working(self, symbol: str) -> bool:
        return any(
            order.symbol == symbol and order.lifecycle in {"new", "partial"}
            for order in self.orders.values()
        )

    def _bind_identity(self, symbol: str) -> None:
        if symbol in self.symbol_map:
            return
        bind_map(self.symbol_map, symbol, symbol)

    def _drop_unaccepted_mark(self, symbol: str, prior: float | None) -> None:
        """A denied order must not leave the first print behind as a reference."""
        if prior is not None:
            return
        self.marks.pop(symbol, None)
        self.mark_ts.pop(symbol, None)

    def _require_reference(self, snapshot: ReferenceSnapshot) -> None:
        if snapshot.version < 1 or snapshot.max_age_ns < 0 or snapshot.as_of_ns < 0:
            raise LiveDeskError("bad_reference_version")
        if set(snapshot.lot_size) != set(snapshot.tick_size):
            raise LiveDeskError("bad_reference")
        for name in snapshot.lot_size:
            if price_problem(snapshot.lot_size[name]) or price_problem(snapshot.tick_size[name]):
                raise LiveDeskError("bad_reference")

    def _mark_is_stale(self, symbol: str, ts_ns: int) -> bool:
        prior = self.mark_ts.get(symbol)
        if prior is None:
            return False
        if ts_ns < prior:
            return True
        return ts_ns > prior + int(self.config.limits.max_mark_age_ns)

    def _quote_is_older(self, symbol: str, ts_ns: int) -> bool:
        prior = self.mark_ts.get(symbol)
        return prior is not None and ts_ns < prior

    def _going_backwards(self, ts_ns: int) -> bool:
        last = int(self.engine.book.last_ts)
        return last != 0 and ts_ns < last

    def _clock(self, ts_ns: int) -> None:
        if ts_ns < 0:
            raise LiveDeskError("ts_ns must be non-negative")

    def _require_funded(self) -> None:
        if not self.funded:
            raise LiveDeskError(
                "fund the desk with --cash and --account before arming or submitting"
            )

    def _latch(self, reason: str) -> None:
        if not self.engine.book.killed:
            stamp = int(self.engine.book.last_ts)
            self.engine.trip("desk", reason, stamp if stamp > 0 else 0)
        if not self.halt_reason:
            self.halt_reason = reason

    def _freeze(self, reason: str) -> None:
        self._integrity_frozen = True
        self.trace_ok = False
        self._latch(reason)

    def _alive(self) -> None:
        if self._closed:
            raise LiveDeskError("desk is closed")

    def _account_value(self) -> float:
        value = self.cash
        book = self.engine.book
        idx = int(book.uc_tail)
        for _ in range(int(book.uc_live)):
            value += float(book.uc_amt[idx])
            idx = (idx + 1) & int(book.uc_mask)
        for symbol, qty in self.positions.items():
            mark = self.marks.get(symbol)
            if mark is not None and mark == mark:
                value += qty * mark
        return value

    def _data_label(self) -> str:
        if self.arming.venue == "loopback":
            return "SYNTHETIC"
        if self.arming.venue == "drop":
            return "OPERATOR_BRIDGE"
        return "SHADOW"

    def _envelope(self, **extra: object) -> dict[str, object]:
        orders = [self.orders[oid].as_dict() for oid in self.order_ids]
        body: dict[str, object] = {
            "schema": "live_desk_report.v1",
            "root": str(self.root),
            "venue": self.arming.venue,
            "transmission": self._transmits(),
            "held_reason": held_reason(self.arming, os.environ),
            "evidence_conditions": "blocked",
            "live_pnl_claim": False,
            "would_promote_live": False,
            "data_label": self._data_label(),
            "kill_latched": bool(self.engine.book.killed) or self._integrity_frozen,
            "kill_reasons": list(reason_names(int(self.engine.book.kill_reason))),
            "halt_reason": self.halt_reason,
            "cash": self.cash,
            "account_value": self._account_value(),
            "positions": {key: value for key, value in self.positions.items()},
            "reserved_cash": self.reserved_cash,
            "funded": self.funded,
            "trace_ok": self.trace_ok and not self._integrity_frozen,
            "orders": orders,
            "orphan_count": len(self.orphans),
        }
        body.update(extra)
        return body

    def _journal(self, kind: str, payload: Mapping[str, object]) -> None:
        if self._integrity_frozen:
            raise LiveDeskError(self.halt_reason or "journal integrity failure")
        body = {"kind": kind, "payload": payload, "prev": self.journal_hash}
        digest = _digest(body)
        row = {"event_hash": digest, "kind": kind, "payload": payload, "prev": self.journal_hash}
        _append_line(self.root / "journal.jsonl", row)
        self.journal_hash = digest

    def _save(self) -> None:
        if self._integrity_frozen:
            raise LiveDeskError(self.halt_reason or "journal integrity failure")
        self._check_books()
        _atomic_json(self.root / "state.json", self._snapshot())

    def _snapshot(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "config_sha256": self.config_sha256,
            "cash": self.cash,
            "account_peak": self.peak,
            "session_start": self.session_start,
            "positions": dict(self.positions),
            "marks": dict(self.marks),
            "mark_ts": dict(self.mark_ts),
            "reserved_qty": dict(self.reserved_qty),
            "reserved_cash": self.reserved_cash,
            "orders": [self.orders[oid].as_dict() for oid in self.order_ids],
            "trace": list(self.trace),
            "seen_events": sorted(self.seen),
            "orphans": list(self.orphans),
            "journal_hash": self.journal_hash,
            "last_ts": int(self.engine.book.last_ts),
            "halt_reason": self.halt_reason,
            "kill_latched": bool(self.engine.book.killed),
            "kill_reason": int(self.engine.book.kill_reason),
            "trace_ok": self.trace_ok,
            "rate_order_ts": list(self.rate_order_ts),
            "rate_message_ts": list(self.rate_message_ts),
            "dupes": [list(row) for row in self.dupes],
            "lots": _dump_lots(self.engine),
            "unsettled": _dump_unsettled(self.engine),
            "day_trades": [
                int(self.engine.book.dt_sess[i]) for i in range(int(self.engine.book.dt_n))
            ],
            "opened_sessions": _dump_opened(self.engine),
            "live_pnl_claim": False,
            "would_promote_live": False,
            "evidence_conditions": "blocked",
            "symbol_controls": {
                symbol: {"halted": halt, "locate_ok": locate}
                for symbol, (halt, locate) in self.controls.items()
            },
            "limit_override": _limit_payload(self._limit_override),
            "symbol_map": dict(self.symbol_map),
            "feeds": {
                symbol: {"seq": seq, "stale": stale} for symbol, (seq, stale) in self.feeds.items()
            },
            "reference": None if self.reference is None else self.reference.as_dict(),
            "corporate_actions": list(self.actions),
            "action_halt": sorted(self.action_halt),
            "quote_ok": dict(self.quote_ok),
            "price_bands": {
                symbol: {"lower": lower, "upper": upper}
                for symbol, (lower, upper) in self.bands.items()
            },
        }

    def _load_without_snapshot(
        self,
        path: Path,
        *,
        cash: float | None,
        account_value: float | None,
        ts_ns: int,
    ) -> bool:
        """Start a desk that has no ``state.json``. True when that path applied."""
        if path.exists():
            return False
        if (self.root / "journal.jsonl").exists():
            self._freeze("journal exists without a desk snapshot")
            return True
        if cash is None or account_value is None:
            self.funded = False
            self.cash = 0.0
            return True
        self.funded = True
        self.cash = _positive(cash, None)
        self.engine.book.settled = self.cash
        value = _positive(account_value, None)
        self.peak = value
        self.session_start = value
        self.engine.update_account(nav=value, mark_ts_ns=ts_ns if ts_ns > 0 else 0)
        return True

    def _restore_orders(self, raw: Mapping[str, Any]) -> None:
        self.orders = {}
        self.order_ids = []
        self._by_client = {}
        for item in list(raw["orders"]):
            if not isinstance(item, dict):
                raise LiveDeskError("order record must be an object")
            order = _order_from(item)
            self._store(order)

    def _restore_kill(self, raw: Mapping[str, Any]) -> None:
        if bool(raw.get("kill_latched")):
            if not self.engine.book.killed:
                self.engine.trip(
                    "desk", self.halt_reason or "restored halt", int(raw.get("last_ts") or 0)
                )
            self.engine.book.kill_reason |= int(raw.get("kill_reason") or 0)

    def _load(self, *, cash: float | None, account_value: float | None, ts_ns: int) -> None:
        path = self.root / "state.json"
        if self._load_without_snapshot(path, cash=cash, account_value=account_value, ts_ns=ts_ns):
            return
        if cash is not None or account_value is not None:
            raise LiveDeskError("cash and account are already persisted; refusing to override")
        try:
            text = path.read_text(encoding="utf-8")
            raw = json.loads(text) if text.strip() else None
        except json.JSONDecodeError as exc:
            self._latch("desk state file is truncated")
            raise LiveDeskError("desk state file is truncated") from exc
        if not isinstance(raw, dict):
            self._latch("desk state file is corrupt")
            raise LiveDeskError("desk state file is corrupt")
        _require_closed_state(raw)
        if raw.get("config_sha256") != self.config_sha256:
            self._latch("risk config changed")
        self.funded = True
        self.cash = float(raw["cash"])
        self.peak = float(raw["account_peak"])
        self.session_start = float(raw["session_start"])
        self.positions = {str(k): float(v) for k, v in dict(raw["positions"]).items()}
        self.marks = {str(k): float(v) for k, v in dict(raw["marks"]).items()}
        self.mark_ts = {str(k): int(v) for k, v in dict(raw["mark_ts"]).items()}
        self.reserved_qty = {str(k): float(v) for k, v in dict(raw["reserved_qty"]).items()}
        self.reserved_cash = float(raw["reserved_cash"])
        self.journal_hash = str(raw["journal_hash"])
        self.halt_reason = str(raw.get("halt_reason") or "")
        self.trace_ok = bool(raw.get("trace_ok", True))
        self.trace = [dict(event) for event in list(raw.get("trace", []))]
        self.seen = {str(item) for item in list(raw.get("seen_events", []))}
        self.orphans = [dict(item) for item in list(raw.get("orphans", []))]
        self.rate_order_ts = [int(item) for item in list(raw.get("rate_order_ts", []))]
        self.rate_message_ts = [int(item) for item in list(raw.get("rate_message_ts", []))]
        self.dupes = [_dupe(item) for item in list(raw.get("dupes", []))]
        self._restore_orders(raw)
        self._restore_market(raw)
        self._restore_engine(raw)
        self._restore_controls(raw)
        self._restore_saved_limits(raw)
        self._restore_trace()
        problem = self._verify_journal()
        if problem:
            self._freeze(problem)
            return
        if raw.get("config_sha256") != self.config_sha256:
            self._latch("risk config changed")
        self._scan_outbound()
        self._restore_kill(raw)
        self.engine.book.last_ts = int(raw.get("last_ts") or 0)

    def _restore_lots(self, raw: Mapping[str, Any]) -> None:
        for symbol, lots in dict(raw.get("lots", {})).items():
            sid = self._sid(str(symbol))
            if not isinstance(lots, list):
                raise LiveDeskError("lots must be a list")
            for lot in lots:
                if not isinstance(lot, dict):
                    raise LiveDeskError("lot must be an object")
                self.engine.restore_lot(
                    sid,
                    qty=float(lot["qty"]),
                    settles_ns=int(lot["settles_ns"]),
                    unsettled_funded=bool(lot["unsettled"]),
                )

    def _restore_opened_sessions(self, raw: Mapping[str, Any]) -> None:
        opened = raw.get("opened_sessions", {})
        if isinstance(opened, dict):
            for symbol, sess in opened.items():
                sid = self.engine.ensure_symbol(str(symbol))
                if sid >= 0:
                    self.engine.book.syms[sid].opened_session = int(sess)

    def _restore_unsettled_cash(self, raw: Mapping[str, Any]) -> None:
        unsettled = raw.get("unsettled", [])
        if isinstance(unsettled, list):
            for item in unsettled:
                if not isinstance(item, dict):
                    raise LiveDeskError("unsettled cash must be an object")
                if not self.engine._enqueue_unsettled(
                    float(item["amount"]), int(item["available_ns"])
                ):
                    self._latch("unsettled cash queue failed to restore")

    def _restore_day_trades(self, raw: Mapping[str, Any]) -> None:
        trades = raw.get("day_trades", [])
        if isinstance(trades, list):
            for sess_id in trades:
                if self.engine.book.dt_n >= self.engine.book.dt_cap:
                    self._latch("day-trade memory failed to restore")
                    break
                self.engine.book.dt_sess[self.engine.book.dt_n] = int(sess_id)
                self.engine.book.dt_n += 1

    def _restore_engine(self, raw: Mapping[str, Any]) -> None:
        self.engine.book.settled = self.cash
        self.engine.book.peak = self.peak
        self.engine.book.session_start = self.session_start
        self.engine.update_account(
            nav=self._account_value(),
            mark_ts_ns=0,
            session_start_nav=self.session_start,
            peak_nav=self.peak,
        )
        self._restore_lots(raw)
        self._restore_opened_sessions(raw)
        self._restore_unsettled_cash(raw)
        self._restore_day_trades(raw)

    def _restore_trace(self) -> None:
        checked = check_trace(self.trace)
        self.book = checked.book
        if not checked.ok:
            self.trace_ok = False
            self._latch("stored lifecycle trace failed verification")
            return
        for order in self.orders.values():
            if order.lifecycle == "absent":
                continue
            state = self.book.orders.get(order.order_id)
            if state is None or state.status != order.lifecycle:
                self.trace_ok = False
                self._latch(f"order {order.order_id} does not match the lifecycle trace")
                return

    def _verify_journal(self) -> str | None:
        path = self.root / "journal.jsonl"
        if not path.exists():
            if self.journal_hash != _ZERO:
                return "journal missing"
            return None
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return f"journal unreadable: {exc}"
        if lines and _torn_json(lines[-1]):
            lines = lines[:-1]
            _rewrite_lines(path, lines)
        return _journal_problem(lines, self.journal_hash)

    def _scan_outbound(self) -> None:
        outbound = self.root / "outbound"
        if not outbound.exists():
            return
        for path in sorted(outbound.glob("*.json")):
            if path.name.startswith("cancel-") or path.name.startswith("."):
                continue
            if path.stem not in self.orders:
                self.orphans.append({"order_id": path.stem, "event_id": "outbound"})
                self._latch(f"outbound ticket {path.name} has no desk order")

    def _acquire(self) -> None:
        path = self.root / ".lock"
        for _ in range(2):
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if _lock_alive(path):
                    raise LiveDeskError(f"desk is locked: {path}") from None
                try:
                    path.unlink()
                except OSError as exc:
                    raise LiveDeskError(f"desk is locked: {path}") from exc
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self._lock_path = path
            return
        raise LiveDeskError(f"desk is locked: {path}")


def _action_record(kind: str, plan: ActionPlan, ts_ns: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "symbol": plan.symbol,
        "new_symbol": plan.new_symbol,
        "apply": plan.apply,
        "halt": plan.halt,
        "reason": plan.reason,
        "qty_factor": plan.qty_factor,
        "cash_delta": plan.cash_delta,
        "ts_ns": ts_ns,
    }


def _action_list(raw: object) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LiveDeskError("corporate actions are malformed")
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise LiveDeskError("corporate action is malformed")
        rows.append(
            {
                "kind": str(item.get("kind", "")),
                "symbol": str(item["symbol"]),
                "new_symbol": str(item.get("new_symbol", item["symbol"])),
                "apply": bool(item.get("apply")),
                "halt": bool(item.get("halt")),
                "reason": str(item.get("reason", "")),
                "qty_factor": float(item.get("qty_factor", 1.0)),
                "cash_delta": float(item.get("cash_delta", 0.0)),
                "ts_ns": int(item.get("ts_ns", 0)),
            }
        )
    return rows


def _feed_map(raw: object) -> dict[str, tuple[int | None, bool]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise LiveDeskError("feed state is malformed")
    feeds: dict[str, tuple[int | None, bool]] = {}
    for symbol, item in raw.items():
        if not isinstance(item, dict):
            raise LiveDeskError("feed state is malformed")
        seq = item.get("seq")
        feeds[str(symbol)] = (None if seq is None else int(seq), bool(item.get("stale")))
    return feeds


def _reference_from_state(raw: object) -> ReferenceSnapshot | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise LiveDeskError("reference snapshot is malformed")
    try:
        return snapshot_from_dict({str(key): value for key, value in raw.items()})
    except (MarketDataError, KeyError, TypeError, ValueError) as exc:
        raise LiveDeskError("reference snapshot is malformed") from exc


def _valid_band(lower: float, upper: float) -> tuple[float, float]:
    lo = float(lower)
    hi = float(upper)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0.0 or hi < lo:
        raise LiveDeskError("bad_price_band")
    return lo, hi


def _positive(value: float | None, default: float | None) -> float:
    chosen = default if value is None else value
    if chosen is None or chosen != chosen or chosen <= 0.0 or chosen == float("inf"):
        raise LiveDeskError("cash and account must be finite and positive")
    return float(chosen)


def _symbol(symbol: str) -> str:
    text = symbol.strip().upper()
    if not text or len(text) > 32 or any(ch not in _SYMBOL_OK for ch in text):
        raise LiveDeskError(f"symbol {symbol!r} is not a ticker")
    return text


def _side(side: str) -> str:
    text = side.strip().upper()
    if text not in {"BUY", "SELL"}:
        raise LiveDeskError("side must be BUY or SELL")
    return text


def _merge_reasons(reasons: list[str], extra: tuple[str, ...]) -> None:
    for reason in extra:
        if reason not in reasons:
            reasons.append(reason)


def _client(client: str) -> None:
    if not client or len(client) > 64 or any(ch not in _CLIENT_OK for ch in client):
        raise LiveDeskError("client_order_id must be 1-64 characters of letters, digits, . _ -")


def _quantity(qty: float) -> None:
    if isinstance(qty, bool) or qty != qty or qty <= 0.0 or qty == float("inf"):
        raise LiveDeskError("quantity must be finite and positive")


def _price(px: float) -> None:
    if isinstance(px, bool) or px != px or px <= 0.0 or px == float("inf"):
        raise LiveDeskError("price must be finite and positive")


def _limit_payload(limits: LimitConfig | None) -> dict[str, object] | None:
    if limits is None:
        return None
    return limits.model_dump(mode="json")


def _ticks(
    config: PretradeConfig, symbol: str, side: str, qty: float, px: float
) -> tuple[str, str, int, int]:
    qt = int(round(qty / float(config.limits.qty_tick)))
    pt = int(round(px / float(config.limits.price_tick)))
    return symbol, side, qt, pt


def _ticket(order: DeskOrder) -> dict[str, object]:
    return {
        "schema": "live_order_ticket.v1",
        "ticket": "order",
        "order_id": order.order_id,
        "client_order_id": order.client_order_id,
        "symbol": order.symbol,
        "side": order.side,
        "qty": order.qty,
        "limit_price": order.px,
        "live_pnl_claim": False,
        "would_promote_live": False,
        "evidence_conditions": "blocked",
    }


def _cancel_ticket(order: DeskOrder) -> dict[str, object]:
    return {
        "schema": "live_cancel_ticket.v1",
        "ticket": "cancel",
        "order_id": order.order_id,
        "symbol": order.symbol,
        "live_pnl_claim": False,
        "would_promote_live": False,
        "evidence_conditions": "blocked",
    }


def _digest(body: Mapping[str, object]) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _journal_problem(lines: list[str], expected: str) -> str | None:
    """Walk the chain. A torn last line is dropped; a bad earlier line still freezes."""
    body = list(lines)
    if body and _torn_json(body[-1]):
        body = body[:-1]
    prev = _ZERO
    last = _ZERO
    for line_no, line in enumerate(body, start=1):
        digest, problem = _journal_link(line, line_no, prev)
        if problem is not None:
            return problem
        prev = digest
        last = digest
    if last != expected:
        return "journal head does not match the desk snapshot"
    return None


def _torn_json(line: str) -> bool:
    try:
        json.loads(line)
    except json.JSONDecodeError:
        return True
    return False


def _rewrite_lines(path: Path, lines: list[str]) -> None:
    """Drop a torn tail so the next append chains from the last intact line."""
    payload = "".join(f"{line}\n" for line in lines).encode()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _journal_link(line: str, line_no: int, prev: str) -> tuple[str, str | None]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return "", f"journal break at line {line_no}"
    if not isinstance(row, dict) or row.get("prev") != prev:
        return "", f"journal break at line {line_no}"
    digest = _digest({"kind": row.get("kind"), "payload": row.get("payload"), "prev": prev})
    if row.get("event_hash") != digest:
        return "", f"journal hash mismatch at line {line_no}"
    return digest, None


def _append_line(path: Path, row: Mapping[str, object]) -> None:
    line = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _require_closed_state(raw: Mapping[str, Any]) -> None:
    if raw.get("schema") != _SCHEMA:
        raise LiveDeskError("desk state schema is not live_desk_state.v1")
    if raw.get("live_pnl_claim") is not False:
        raise LiveDeskError("desk state must set live_pnl_claim false")
    if raw.get("would_promote_live") is not False:
        raise LiveDeskError("desk state must set would_promote_live false")
    if raw.get("evidence_conditions") != "blocked":
        raise LiveDeskError("evidence conditions stay blocked")


def _order_from(raw: Mapping[str, Any]) -> DeskOrder:
    reasons = raw.get("reasons", [])
    if not isinstance(reasons, list):
        raise LiveDeskError("order reasons must be a list")
    return DeskOrder(
        order_id=str(raw["order_id"]),
        client_order_id=str(raw["client_order_id"]),
        symbol=str(raw["symbol"]),
        side=str(raw["side"]),
        qty=float(raw["qty"]),
        px=float(raw["px"]),
        filled_qty=float(raw.get("filled_qty") or 0.0),
        status=str(raw.get("status") or "working"),
        lifecycle=str(raw.get("lifecycle") or "absent"),
        venue=str(raw.get("venue") or "shadow"),
        synthetic=bool(raw.get("synthetic")),
        transmitted=bool(raw.get("transmitted")),
        acked=bool(raw.get("acked")),
        reasons=tuple(str(item) for item in reasons),
    )


def _dupe(item: object) -> tuple[str, str, int, int, int]:
    if not isinstance(item, list) or len(item) != 5:
        raise LiveDeskError("duplicate memory must be a 5-item list")
    return (str(item[0]), str(item[1]), int(item[2]), int(item[3]), int(item[4]))


def _dump_lots(engine: PretradeEngine) -> dict[str, list[dict[str, object]]]:
    dumped: dict[str, list[dict[str, object]]] = {}
    for symbol, sid in engine._ids.items():
        slot = engine.book.syms[sid]
        lots: list[dict[str, object]] = []
        for index in range(int(slot.nlot)):
            lots.append(
                {
                    "qty": float(slot.lq[index]),
                    "settles_ns": int(slot.ls[index]),
                    "unsettled": int(slot.lu[index]),
                }
            )
        if lots:
            dumped[symbol] = lots
    return dumped


def _dump_unsettled(engine: PretradeEngine) -> list[dict[str, object]]:
    book = engine.book
    rows: list[dict[str, object]] = []
    idx = int(book.uc_tail)
    for _ in range(int(book.uc_live)):
        rows.append({"amount": float(book.uc_amt[idx]), "available_ns": int(book.uc_ts[idx])})
        idx = (idx + 1) & int(book.uc_mask)
    return rows


def _dump_opened(engine: PretradeEngine) -> dict[str, int]:
    opened: dict[str, int] = {}
    for symbol, sid in engine._ids.items():
        sess = int(engine.book.syms[sid].opened_session)
        if sess:
            opened[symbol] = sess
    return opened


def _lock_alive(path: Path) -> bool:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        pid = int(raw)
    except (OSError, ValueError):
        return True
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
