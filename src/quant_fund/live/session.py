"""Armed order session.

Orders reach a broker only after the arm file matches this config and the
pre-trade engine allows them. Reports are broker accounting. ``live_pnl_claim``
is always false; nothing here is a research receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn
from zoneinfo import ZoneInfo

from quant_fund.live.brokers import (
    AlpacaBroker,
    Broker,
    BrokerAccount,
    BrokerAck,
    BrokerFill,
    LiveOrder,
    LoopbackBroker,
    alpaca_credentials,
    map_broker_status,
)
from quant_fund.live.config import LiveConfig, load_gateway_pretrade
from quant_fund.live.errors import BrokerRejected, LiveGateError
from quant_fund.live.exchange_clock import (
    ClockError,
    band_map,
    flag_map,
    move_entry,
    nyse_rth_closed,
    outside_band,
)
from quant_fund.live.fat_finger import fat_finger_reasons
from quant_fund.live.gateway_arm import (
    arm_session,
    disarm_session,
    load_hmac_key,
    read_arm,
    zero_audit_head,
)
from quant_fund.live.idempotency import assign_key
from quant_fund.live.ledger import LedgerFill, LiveLedger, ledger_fill_from_dict
from quant_fund.live.market_data import (
    ActionPlan,
    MarketDataError,
    ReferenceSnapshot,
    assess_quote,
    jump_problem,
    mapping_is_bijective,
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
from quant_fund.live.risk_memory import export_pretrade, restore_pretrade
from quant_fund.pretrade.codes import KIND_CANCEL, KIND_ORDER
from quant_fund.pretrade.config import LimitConfig, PretradeConfig, sign_config
from quant_fund.pretrade.engine import OrderView, PretradeEngine

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_CLIENT = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
_ORDER_STATUSES = frozenset(
    {
        "inflight",
        "accepted",
        "filled",
        "partially_filled",
        "canceled",
        "rejected",
        "not_acknowledged",
        "expired",
        "replaced",
        "done_for_day",
        "pending_cancel",
        "pending_replace",
        "suspended",
        "unmapped",
    }
)
_TERMINAL = frozenset(
    {
        "filled",
        "canceled",
        "rejected",
        "not_acknowledged",
        "expired",
        "replaced",
        "done_for_day",
        "suspended",
    }
)
_WORKING = frozenset(
    {
        "inflight",
        "accepted",
        "partially_filled",
        "pending_cancel",
        "pending_replace",
        "unmapped",
    }
)
_STATE_SCHEMA = "live_session.v1"
_RETRY_BUDGET = 3
_RETRY_DELAYS = (0.01, 0.02)
_RETRY_HTTP = ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")


def _limit_payload(limits: LimitConfig | None) -> dict[str, Any] | None:
    if limits is None:
        return None
    dumped: dict[str, Any] = limits.model_dump(mode="json")
    return dumped


def _valid_band(lower: float, upper: float) -> tuple[float, float]:
    lo = float(lower)
    hi = float(upper)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0.0 or hi < lo:
        raise LiveGateError("bad_price_band")
    return lo, hi


def _market_reply(**fields: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": True,
        "live_pnl_claim": False,
        "research_evidence": False,
    }
    body.update(fields)
    return body


def _action_record(kind: str, plan: ActionPlan, ts: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "symbol": plan.symbol,
        "new_symbol": plan.new_symbol,
        "apply": plan.apply,
        "halt": plan.halt,
        "reason": plan.reason,
        "qty_factor": plan.qty_factor,
        "cash_delta": plan.cash_delta,
        "ts_ns": ts,
    }


def _action_reply(record: dict[str, Any]) -> dict[str, Any]:
    body = dict(record)
    body["ok"] = bool(record["apply"])
    body["live_pnl_claim"] = False
    body["research_evidence"] = False
    return body


def _action_list(raw: object) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LiveGateError("corporate actions are malformed")
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise LiveGateError("corporate action is malformed")
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
        raise LiveGateError("feed state is malformed")
    feeds: dict[str, tuple[int | None, bool]] = {}
    for symbol, item in raw.items():
        if not isinstance(item, dict):
            raise LiveGateError("feed state is malformed")
        seq = item.get("seq")
        feeds[str(symbol)] = (None if seq is None else int(seq), bool(item.get("stale")))
    return feeds


def _reference_from_state(raw: object) -> ReferenceSnapshot | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise LiveGateError("reference snapshot is malformed")
    try:
        return snapshot_from_dict(raw)
    except (MarketDataError, KeyError, TypeError, ValueError) as exc:
        raise LiveGateError("reference snapshot is malformed") from exc


def _limit_reply(
    *,
    applied: bool,
    loosened: tuple[str, ...] = (),
    structural: tuple[str, ...] = (),
    reauthorized: bool = False,
) -> dict[str, Any]:
    return {
        "ok": applied,
        "applied": applied,
        "loosened": list(loosened),
        "structural": list(structural),
        "reauthorized": reauthorized,
        "live_pnl_claim": False,
        "research_evidence": False,
    }


def _retry_sleep(delay: float) -> None:
    """Backoff between transient broker attempts. Tests replace this."""
    time.sleep(delay)


def _retry_delay(attempt: int) -> float:
    index = attempt if attempt < len(_RETRY_DELAYS) else len(_RETRY_DELAYS) - 1
    return _RETRY_DELAYS[index]


def _retryable_reject(exc: BrokerRejected) -> bool:
    text = str(exc)
    return any(code in text for code in _RETRY_HTTP)


def _rejected_ack(order: LiveOrder, reason: str) -> BrokerAck:
    return BrokerAck(
        client_order_id=order.client_order_id,
        broker_order_id="",
        status="rejected",
        filled_quantity=0.0,
        fill_price=None,
        fills=(),
        reason=reason,
    )


def _actor(actor: str) -> str:
    cleaned = actor.strip()
    if not cleaned or len(cleaned) > 80 or any(ord(ch) < 32 for ch in cleaned):
        raise LiveGateError("actor must be 1..80 visible characters")
    return cleaned


def _reason(reason: str) -> str:
    cleaned = reason.strip()
    if not cleaned or len(cleaned) > 200:
        raise LiveGateError("reason must be 1..200 characters")
    return cleaned


def _symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not _SYMBOL.fullmatch(cleaned):
        raise LiveGateError("symbol must be 1..10 letters, digits, or dots")
    return cleaned


def _side(side: str) -> str:
    cleaned = side.strip().lower()
    if cleaned not in {"buy", "sell"}:
        raise LiveGateError("side must be buy or sell")
    return cleaned


def _positive(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise LiveGateError(f"{label} must be finite and positive")
    return number


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _read_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LiveGateError("session state file is unreadable") from exc
    if not text.strip():
        raise LiveGateError("session state file is truncated")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LiveGateError("session state file is truncated") from exc
    if not isinstance(raw, dict):
        raise LiveGateError("session state file is corrupt")
    return raw


class LiveSession:
    """One state directory, one venue, one pre-trade snapshot."""

    def __init__(
        self,
        config: LiveConfig,
        *,
        state_dir: Path,
        pretrade: PretradeConfig,
        pretrade_sha256: str,
        broker: Broker,
        now_ns: Callable[[], int],
    ) -> None:
        self.config = config
        self.venue = config.venue
        self.state_dir = state_dir
        self.pretrade = pretrade
        self.pretrade_sha256 = pretrade_sha256
        self.broker = broker
        self._now = now_ns
        self.orders: list[dict[str, Any]] = []
        self.marks: dict[str, float] = {}
        self.kill_state: dict[str, Any] = {"latched": False, "actor": "", "reason": ""}
        self._halted: dict[str, bool] = {}
        self._locate: dict[str, bool] = {}
        self._limit_override: LimitConfig | None = None
        self._symbol_map: dict[str, str] = {}
        self._feeds: dict[str, tuple[int | None, bool]] = {}
        self._reference: ReferenceSnapshot | None = None
        self._actions: list[dict[str, Any]] = []
        self._action_halt: set[str] = set()
        self._quote_ok: dict[str, bool] = {}
        self._bands: dict[str, tuple[float, float]] = {}
        self._thread_lock = threading.Lock()
        self.ledger = LiveLedger(
            baseline_cash=float(config.starting_cash),
            cash_tolerance=config.cash_tolerance,
            position_tolerance=config.position_tolerance,
        )
        self._baseline_from_broker = config.venue == "loopback"
        self.arm_path = state_dir / "arm.json"
        self.state_path = state_dir / "state.json"
        self.audit_path = state_dir / "audit.jsonl"
        self._arm: dict[str, Any] | None = None
        self.audit_head = zero_audit_head()
        self._sids: dict[str, int] = {}
        self._engine_noted: set[str] = set()
        self._pretrade_blob: object = None
        self._model_settled = float(config.starting_cash)
        self._disk_token: str | None = None
        self.engine = PretradeEngine(
            pretrade,
            hmac_key=_hmac_for_engine(state_dir),
            initial_nav=float(config.starting_cash),
            initial_cash=float(config.starting_cash),
            ts_ns=0,
        )
        # State and the audit head are read only after the interprocess lock
        # is held. A second CLI process cannot overwrite this snapshot.
        with self._lock():
            pass

    def status(self) -> dict[str, Any]:
        """Local snapshot. This does not place an order."""
        with self._lock():
            return self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        return {
            "ok": True,
            "venue": self.venue,
            "calendar": self.config.calendar,
            "armed": self._arm is not None,
            "actor": None if self._arm is None else self._arm.get("actor"),
            "kill_latched": bool(self.kill_state["latched"]),
            "credentials_present": _credentials_present(self.broker),
            "orders": len(self.orders),
            "computed_cash": self.ledger.computed_cash,
            "computed_positions": self.ledger.computed_positions,
            "marks": dict(self.marks),
            "live_pnl_claim": False,
            "research_evidence": False,
        }

    def arm(self, *, actor: str, confirm_live_capital: str = "") -> dict[str, Any]:
        with self._lock():
            payload = arm_session(
                self.arm_path,
                venue=self.venue,
                actor=actor,
                pretrade_sha256=self.pretrade_sha256,
                confirm_live_capital=confirm_live_capital,
            )
            self._arm = payload
            self._audit("arm", {"actor": payload["actor"], "venue": self.venue})
            return {
                "ok": True,
                "armed": True,
                "venue": self.venue,
                "actor": payload["actor"],
                "live_capital_confirmed": payload["live_capital_confirmed"],
                "live_pnl_claim": False,
                "research_evidence": False,
            }

    def disarm(self, *, actor: str) -> dict[str, Any]:
        with self._lock():
            result = disarm_session(self.arm_path, actor=actor)
            self._arm = None
            self._audit("disarm", {"actor": result["actor"]})
            return result

    def kill(self, *, actor: str, reason: str, at_ns: int | None = None) -> dict[str, Any]:
        with self._lock():
            who = _actor(actor)
            why = _reason(reason)
            ts = self._clock(at_ns)
            self.engine.trip(who, why, ts)
            self.kill_state = {"latched": True, "actor": who, "reason": why}
            canceled = self._cancel_working(ts)
            self._persist()
            self._audit("kill", {"actor": who, "reason": why, "canceled": canceled})
            still = self._working_ids()
            return {
                "ok": not still,
                "kill_latched": True,
                "canceled": canceled,
                "still_working": still,
                "live_pnl_claim": False,
                "research_evidence": False,
            }

    def reset_kill(self, *, actor: str, reason: str, at_ns: int | None = None) -> dict[str, Any]:
        with self._lock():
            who = _actor(actor)
            why = _reason(reason)
            ts = self._clock(at_ns)
            self.engine.reset(who, why, ts)
            self.kill_state = {"latched": False, "actor": who, "reason": why}
            self._persist()
            self._audit("reset", {"actor": who, "reason": why})
            return {
                "ok": True,
                "kill_latched": False,
                "live_pnl_claim": False,
                "research_evidence": False,
            }

    def mark(
        self, *, symbol: str, price: float, actor: str, at_ns: int | None = None
    ) -> dict[str, Any]:
        with self._lock():
            self._require_trading()
            _actor(actor)
            cleaned = _symbol(symbol)
            number = _positive(price, "mark")
            ts = self._clock(at_ns)
            if self._quote_moves_backward(cleaned, ts):
                raise LiveGateError("mark timestamp is older than the stored quote")
            problem = jump_problem(self.marks.get(cleaned), number)
            if problem:
                raise LiveGateError(problem)
            self._bind_identity(cleaned)
            self.marks[cleaned] = number
            self.broker.set_mark(cleaned, number)
            self._sync_account(ts)
            self._enforce_kill(ts)
            self._persist()
            return {
                "ok": True,
                "symbol": cleaned,
                "mark": number,
                "live_pnl_claim": False,
                "research_evidence": False,
            }

    def bind_symbol(self, *, internal: str, broker: str, actor: str) -> dict[str, Any]:
        """Bind one internal id to one broker symbol. A second binding is an error."""
        with self._lock():
            self._require_trading()
            _actor(actor)
            left = _symbol(internal)
            try:
                bind_map(self._symbol_map, left, broker)
            except MarketDataError as exc:
                raise LiveGateError(str(exc)) from exc
            self._persist()
            self._audit("symbol_map", {"internal": left, "broker": self._symbol_map[left]})
            return _market_reply(symbol=left, broker=self._symbol_map[left])

    def ingest_quote(
        self,
        *,
        symbol: str,
        bid: float,
        ask: float,
        seq: int,
        actor: str,
        at_ns: int | None = None,
        snapshot: bool = False,
    ) -> dict[str, Any]:
        """Accept a sequenced quote or leave the stored mark untouched."""
        with self._lock():
            self._require_trading()
            _actor(actor)
            return self._ingest_locked(
                symbol=symbol, bid=bid, ask=ask, seq=seq, at_ns=at_ns, snapshot=snapshot
            )

    def set_reference(self, snapshot: ReferenceSnapshot, *, actor: str) -> dict[str, Any]:
        """Install the lot and tick snapshot used on the next order."""
        with self._lock():
            self._require_trading()
            _actor(actor)
            self._require_reference(snapshot)
            self._reference = snapshot
            self._persist()
            self._audit("reference", {"version": snapshot.version})
            return _market_reply(version=snapshot.version)

    def set_price_band(
        self, *, symbol: str, lower: float, upper: float, actor: str
    ) -> dict[str, Any]:
        """Install an inclusive LULD-style band. Prices outside it are not tradable."""
        with self._lock():
            self._require_trading()
            _actor(actor)
            cleaned = _symbol(symbol)
            self._bands[cleaned] = _valid_band(lower, upper)
            self._persist()
            self._audit("price_band", {"symbol": cleaned, "lower": lower, "upper": upper})
            return _market_reply(symbol=cleaned, lower=lower, upper=upper)

    def apply_corporate_action(
        self,
        *,
        kind: str,
        symbol: str,
        actor: str,
        ratio: float | None = None,
        cash_per_share: float | None = None,
        new_symbol: str | None = None,
        at_ns: int | None = None,
    ) -> dict[str, Any]:
        """Adjust one name, or halt it, without changing account value."""
        with self._lock():
            self._require_trading()
            _actor(actor)
            return self._apply_action_locked(
                kind=kind,
                symbol=symbol,
                ratio=ratio,
                cash_per_share=cash_per_share,
                new_symbol=new_symbol,
                at_ns=at_ns,
            )

    def set_symbol_controls(
        self,
        *,
        symbol: str,
        actor: str,
        halted: bool | None = None,
        locate_ok: bool | None = None,
        at_ns: int | None = None,
    ) -> dict[str, Any]:
        """Halt a name or clear its locate. The next order sees it without a restart."""
        with self._lock():
            self._require_trading()
            _actor(actor)
            return self._set_controls_locked(
                symbol=symbol, halted=halted, locate_ok=locate_ok, at_ns=at_ns
            )

    def update_limits(
        self, limits: LimitConfig, *, actor: str, reauthorize: bool = False
    ) -> dict[str, Any]:
        """Apply ceilings now. A looser ceiling needs ``reauthorize``."""
        with self._lock():
            who = _actor(actor)
            self._require_trading()
            return self._update_limits_locked(limits, actor=who, reauthorize=reauthorize)

    def submit(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        mark: float,
        actor: str,
        client_order_id: str | None = None,
        at_ns: int | None = None,
    ) -> dict[str, Any]:
        with self._lock():
            self._prepare_submit(client_order_id=client_order_id)
            return self._submit_locked(
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                mark=mark,
                actor=actor,
                client_order_id=client_order_id,
                at_ns=at_ns,
            )

    def cancel(
        self,
        *,
        client_order_id: str,
        actor: str,
        at_ns: int | None = None,
    ) -> dict[str, Any]:
        with self._lock():
            self._recover_inflight()
            self._require_trading()
            _actor(actor)
            ts = self._clock(at_ns)
            row = self._find(client_order_id)
            if row is None:
                raise LiveGateError("order is not in this session")
            if row["status"] in _TERMINAL:
                return _public_order(row, venue=self.venue, replayed=True)
            if self._quote_is_stale(str(row["symbol"]), ts):
                denied = _denied(
                    reasons=("stale_data",),
                    symbol=str(row["symbol"]),
                    side=str(row["side"]),
                    quantity=float(row["quantity"]),
                    limit_price=float(row["limit_price"]),
                    client_order_id=str(row["client_order_id"]),
                    venue=self.venue,
                )
                self._audit(
                    "deny",
                    {"client_order_id": row["client_order_id"], "reasons": ["stale_data"]},
                )
                return denied
            self._sync_account(ts)
            decision = self.engine.decide(
                self._view(row, float(row["limit_price"]), ts, KIND_CANCEL),
                apply=False,
            )
            if not decision.allowed:
                denied = _denied(
                    reasons=decision.reasons,
                    symbol=str(row["symbol"]),
                    side=str(row["side"]),
                    quantity=float(row["quantity"]),
                    limit_price=float(row["limit_price"]),
                    client_order_id=str(row["client_order_id"]),
                    venue=self.venue,
                )
                self._audit(
                    "deny",
                    {"client_order_id": row["client_order_id"], "reasons": list(decision.reasons)},
                )
                return denied
            try:
                ack = self.broker.cancel(str(row["client_order_id"]), str(row["broker_order_id"]))
            except BrokerRejected as exc:
                row["reason"] = str(exc)
                self._persist()
                return _public_order(row, venue=self.venue, replayed=False)
            self._apply_ack(row, ack, ts)
            self._persist()
            self._audit(
                "cancel", {"client_order_id": row["client_order_id"], "status": row["status"]}
            )
            return _public_order(row, venue=self.venue, replayed=False)

    def reconcile(self, *, at_ns: int | None = None) -> dict[str, Any]:
        with self._lock():
            self._recover_inflight()
            return self._reconcile_locked(at_ns)

    def flatten(self, *, actor: str, at_ns: int | None = None) -> dict[str, Any]:
        """Post a limit at the stored mark for every open position. No market orders."""
        with self._lock():
            self._prepare_submit(client_order_id=None)
            self._require_trading()
            _actor(actor)
            account = self.broker.account()
            results: list[dict[str, Any]] = []
            for symbol, qty in sorted(account.positions.items()):
                if abs(qty) <= 1e-9:
                    continue
                mark = self.marks.get(symbol)
                if mark is None:
                    results.append(
                        {
                            "ok": False,
                            "status": "skipped",
                            "symbol": symbol,
                            "reason": "no_mark",
                            "live_pnl_claim": False,
                            "research_evidence": False,
                        }
                    )
                    continue
                side = "sell" if qty > 0.0 else "buy"
                results.append(
                    self._submit_locked(
                        symbol=symbol,
                        side=side,
                        quantity=abs(qty),
                        limit_price=mark,
                        mark=mark,
                        actor=actor,
                        client_order_id=None,
                        at_ns=at_ns,
                    )
                )
            clean = all(bool(row.get("ok")) for row in results)
            return {
                "ok": clean,
                "orders": results,
                "live_pnl_claim": False,
                "research_evidence": False,
            }

    def _submit_locked(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        mark: float,
        actor: str,
        client_order_id: str | None,
        at_ns: int | None,
    ) -> dict[str, Any]:
        who = _actor(actor)
        ts = self._clock(at_ns)
        self._require_trading()
        cleaned = _symbol(symbol)
        order_side = _side(side)
        qty = _positive(quantity, "quantity")
        limit = _positive(limit_price, "limit price")
        reference = _positive(mark, "mark")
        client = assign_key(
            symbol=cleaned,
            side=order_side,
            quantity=qty,
            limit_price=limit,
            client_order_id=client_order_id,
            known=tuple((str(row["client_order_id"]), str(row["status"])) for row in self.orders),
        )
        if not _CLIENT.fullmatch(client):
            raise LiveGateError("client_order_id must be 1..48 letters, digits, _ or -")
        existing = self._find(client)
        if existing is not None:
            return _public_order(existing, venue=self.venue, replayed=True)
        blocked = self._precision_denial(
            symbol=cleaned,
            side=order_side,
            quantity=qty,
            limit_price=limit,
            client_order_id=client,
            actor=who,
        )
        if blocked is not None:
            return blocked
        finger = self._fat_finger(cleaned, qty, limit, reference)
        self._capture_baseline()
        blocked_market = self._market_block(cleaned, ts, qty, reference)
        if blocked_market:
            reasons = _merge_reasons(blocked_market, finger)
            denied = _denied(
                reasons=reasons,
                symbol=cleaned,
                side=order_side,
                quantity=qty,
                limit_price=limit,
                client_order_id=client,
                venue=self.venue,
            )
            self._audit(
                "deny",
                {"actor": who, "client_order_id": client, "reasons": list(reasons)},
            )
            self._persist()
            return denied
        prior_mark = self.marks.get(cleaned)
        self._bind_identity(cleaned)
        if prior_mark is None:
            self.marks[cleaned] = reference
            self.broker.set_mark(cleaned, reference)
        else:
            self.broker.set_mark(cleaned, prior_mark)
        self._sync_account(ts, reset_session=not self.orders)
        self._enforce_kill(ts)
        placeholder = {
            "client_order_id": client,
            "broker_order_id": "",
            "symbol": cleaned,
            "side": order_side,
            "quantity": qty,
            "limit_price": limit,
            "status": "inflight",
            "filled_quantity": 0.0,
            "fill_price": None,
            "expects_fill": False,
            "reason": "",
            "idempotency_key": client,
        }
        decision = self.engine.decide(self._view(placeholder, limit, ts, KIND_ORDER), apply=False)
        if finger or not decision.allowed:
            self._drop_unaccepted_mark(cleaned, prior_mark)
            reasons = _merge_reasons(decision.reasons, finger)
            denied = _denied(
                reasons=reasons,
                symbol=cleaned,
                side=order_side,
                quantity=qty,
                limit_price=limit,
                client_order_id=client,
                venue=self.venue,
            )
            self._audit("deny", {"actor": who, "client_order_id": client, "reasons": list(reasons)})
            self._persist()
            return denied
        self.marks[cleaned] = reference
        self.broker.set_mark(cleaned, reference)
        self.orders.append(placeholder)
        self.ledger.note_order(client, expects_fill=False, quantity=qty)
        self._persist()
        live_order = LiveOrder(
            client_order_id=client,
            symbol=cleaned,
            side=order_side,
            quantity=qty,
            limit_price=limit,
            fill_time_ns=ts,
        )
        ack = self._send(placeholder, live_order)
        self._apply_ack(placeholder, ack, ts)
        self._enforce_kill(ts)
        self._persist()
        self._audit(
            "submit", {"actor": who, "client_order_id": client, "status": placeholder["status"]}
        )
        return _public_order(placeholder, venue=self.venue, replayed=False)

    def _prepare_submit(self, *, client_order_id: str | None) -> None:
        """Adopt broker fills before a new order when a halt or an inflight order is open."""
        pending = self._open_needs_reconcile()
        self._recover_inflight()
        if self._is_replay(client_order_id):
            return
        if pending or self._open_needs_reconcile() or self._working_has_broker_id():
            self._reconcile_locked(None, ts=self._adoption_clock())

    def _open_needs_reconcile(self) -> bool:
        if self.kill_state.get("latched"):
            return True
        return any(str(row["status"]) == "inflight" for row in self.orders)

    def _is_replay(self, client_order_id: str | None) -> bool:
        if not client_order_id:
            return False
        return self._find(client_order_id) is not None

    def _adoption_clock(self) -> int:
        """Clock for a pre-submit reconcile. Never moves a stored quote backward."""
        stamp = int(self._now())
        mark_ts = int(self.engine.book.mark_ts)
        if mark_ts > stamp:
            stamp = mark_ts
        for sid in self._sids.values():
            sym = self.engine.book.syms[sid]
            if sym.ref_ok and int(sym.ref_ts) > stamp:
                stamp = int(sym.ref_ts)
        return stamp

    def _reconcile_locked(self, at_ns: int | None, *, ts: int | None = None) -> dict[str, Any]:
        """Broker fills page. Caller holds the process lock."""
        self._require_trading()
        stamp = self._clock(at_ns) if ts is None else ts
        self._capture_baseline()
        mapping = {
            str(row["broker_order_id"]): str(row["client_order_id"])
            for row in self.orders
            if row["broker_order_id"]
        }
        page = self.broker.fills(broker_to_client=mapping)
        seen: set[str] = set()
        duplicate_in_batch = False
        ordered = sorted(page.fills, key=lambda fill: (fill.fill_time_ns, fill.fill_id))
        for fill in ordered:
            if fill.fill_id in seen:
                duplicate_in_batch = True
            seen.add(fill.fill_id)
            self._record_fill(fill, stamp)
        account = self.broker.account()
        self.ledger.declare(cash=account.cash, positions=account.positions)
        report = self.ledger.reconcile()
        report["fills_truncated"] = page.truncated
        report["duplicate_in_batch"] = duplicate_in_batch
        if page.truncated or duplicate_in_batch:
            report["ok"] = False
        self._project_fills_onto_orders()
        report["venue"] = self.venue
        alerts = _position_alerts(
            report,
            cash_tolerance=float(self.config.cash_tolerance),
            position_tolerance=float(self.config.position_tolerance),
        )
        report["alerts"] = alerts
        if alerts:
            report["ok"] = False
            self._latch_position_mismatch(alerts, stamp)
            self._audit("alert", {"alerts": alerts})
        self._sync_account(stamp)
        self._enforce_kill(stamp)
        self._persist()
        self._audit("reconcile", {"ok": bool(report["ok"])})
        return report

    def _send(self, row: dict[str, Any], order: LiveOrder) -> BrokerAck:
        for attempt in range(_RETRY_BUDGET):
            ack = self._send_once(row, order)
            if ack is not None:
                return ack
            recovered = self._safe_recover(order.client_order_id)
            if recovered is not None:
                return recovered
            if attempt + 1 >= _RETRY_BUDGET:
                break
            _retry_sleep(_retry_delay(attempt))
        self._lose_response(row)

    def _send_once(self, row: dict[str, Any], order: LiveOrder) -> BrokerAck | None:
        """One broker submit. None means recover, then retry if the budget allows."""
        try:
            return self.broker.submit(order)
        except BrokerRejected as exc:
            if _retryable_reject(exc):
                return None
            return _rejected_ack(order, str(exc))
        except LiveGateError:
            self._mark_unacked(row)
            raise
        except (TimeoutError, ConnectionError):
            return None
        except Exception:
            recovered = self._safe_recover(order.client_order_id)
            if recovered is not None:
                return recovered
            self._lose_response(row)

    def _mark_unacked(self, row: dict[str, Any]) -> None:
        row["status"] = "not_acknowledged"
        row["reason"] = "broker credentials are not set"
        self._persist()

    def _lose_response(self, row: dict[str, Any]) -> NoReturn:
        # A lost response is not proof the broker rejected the order.
        # Leave it inflight so a later open can find it and cancel it.
        row["status"] = "inflight"
        row["reason"] = "broker response lost; order remains recoverable"
        self._persist()
        raise LiveGateError(
            "broker response lost; the order stays inflight until recovery succeeds"
        )

    def _safe_recover(self, client_order_id: str) -> BrokerAck | None:
        try:
            return self.broker.recover(client_order_id)
        except Exception:
            return None

    def _apply_ack(self, row: dict[str, Any], ack: BrokerAck, ts: int) -> None:
        # Fills land before the row becomes terminal, so this ack's prints apply
        # and a later print on the terminal row does not. Filled quantity is
        # updated first so working exposure does not reserve shares already filled.
        row["broker_order_id"] = ack.broker_order_id
        row["filled_quantity"] = float(ack.filled_quantity)
        row["fill_price"] = ack.fill_price
        row["reason"] = ack.reason
        for fill in ack.fills:
            self._record_fill(fill, ts)
        status = map_broker_status(ack.status)
        row["status"] = status
        if status == "unmapped" and not row["reason"]:
            row["reason"] = f"unmapped broker status {ack.status}"
        expects = ack.status in {"filled", "partially_filled"}
        row["expects_fill"] = expects
        self.ledger.set_expects_fill(str(row["client_order_id"]), expects)

    def _record_fill(self, fill: BrokerFill, ts: int) -> str:
        if fill.side not in {"buy", "sell"}:
            raise LiveGateError("broker fill side is not buy or sell")
        fill = self._translate_fill(fill)
        if not self.ledger.has_fill(fill.fill_id):
            row = self._find(fill.client_order_id)
            if row is not None and str(row["status"]) in _TERMINAL:
                self._latch_broker_fault("fill_after_terminal", fill.fill_id, ts)
                return "fill_after_terminal"
        ledger_fill = LedgerFill(
            fill_id=fill.fill_id,
            client_order_id=fill.client_order_id,
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            fill_time_ns=self._fill_stamp(fill, ts),
        )
        verdict = self.ledger.record_fill(ledger_fill)
        if verdict in {"orphan_fill", "overfill"}:
            self._latch_broker_fault(verdict, fill.fill_id, ts)
            return verdict
        if verdict == "duplicate_fill":
            return verdict
        if fill.client_order_id in self._engine_noted or verdict == "replaced_provisional":
            self._sync_account(ts)
        else:
            self._engine_from_fill(fill, ts)
            self._engine_noted.add(fill.client_order_id)
            self._sync_account(ts)
        self._enforce_kill(ts)
        return verdict

    def _fill_stamp(self, fill: BrokerFill, ts: int) -> int:
        """Keep a future print from pushing the ledger watermark past this decision."""
        stamp = int(fill.fill_time_ns) if int(fill.fill_time_ns) > 0 else ts
        horizon = ts + int(self.pretrade.limits.max_mark_age_ns)
        if stamp > horizon:
            return ts
        return stamp

    def _fat_finger(
        self, symbol: str, quantity: float, limit_price: float, mark: float
    ) -> tuple[str, ...]:
        """Caps versus the last stored mark. The first order has no collar yet."""
        anchor = self.marks.get(symbol)
        limits = self.pretrade.limits
        reasons = fat_finger_reasons(
            quantity=quantity,
            limit_price=limit_price,
            mark=anchor if anchor is not None else mark,
            max_quantity=float(limits.max_order_quantity),
            max_notional=float(limits.max_order_notional),
            collar_bps=float(limits.price_collar_bps),
        )
        if anchor is None:
            return tuple(reason for reason in reasons if reason != "price_collar")
        return reasons

    def _latch_restore(self, reason: str) -> None:
        """A bad snapshot halts. Catching the error must not leave trading enabled."""
        text = reason.strip() or "state restore failed"
        self.kill_state = {
            "latched": True,
            "actor": "restore",
            "reason": text[:240],
            "state": "HALT_NEW_ORDERS",
        }
        self.engine.trip("restore", text[:240], int(self._now()))

    def _latch_position_mismatch(self, alerts: list[str], ts: int) -> None:
        if self.kill_state["latched"]:
            return
        reason = "; ".join(alerts)
        self.engine.trip("reconcile", reason, ts)
        self.kill_state = {
            "latched": True,
            "actor": "reconcile",
            "reason": reason,
            "state": "HALT_NEW_ORDERS",
        }

    def _latch_broker_fault(self, verdict: str, fill_id: str, ts: int) -> None:
        """Stop new orders when broker truth is an unknown, excess, or late fill."""
        if self.kill_state["latched"]:
            return
        reason = f"{verdict} {fill_id}"
        self.engine.trip("broker", reason, ts)
        self.kill_state = {"latched": True, "actor": "broker", "reason": reason}

    def _ingest_locked(
        self,
        *,
        symbol: str,
        bid: float,
        ask: float,
        seq: int,
        at_ns: int | None,
        snapshot: bool,
    ) -> dict[str, Any]:
        try:
            cleaned = require_internal(self._symbol_map, symbol)
        except MarketDataError as exc:
            raise LiveGateError(str(exc)) from exc
        ts = self._clock(at_ns)
        if self._nyse_closed(ts):
            self._persist()
            return _market_reply(
                ok=False,
                symbol=cleaned,
                stale=False,
                reasons=["market_closed"],
                mark=self.marks.get(cleaned),
            )
        previous_seq, was_stale = self._feeds.get(cleaned, (None, False))
        decision = assess_quote(
            bid=float(bid),
            ask=float(ask),
            seq=int(seq),
            previous_seq=previous_seq,
            previous_mark=self.marks.get(cleaned),
            was_stale=was_stale,
            snapshot=snapshot,
        )
        accepted, stale, reasons = self._store_quote(cleaned, decision, ts)
        self._persist()
        return _market_reply(
            ok=accepted,
            symbol=cleaned,
            stale=stale,
            reasons=reasons,
            mark=self.marks.get(cleaned),
        )

    def _apply_action_locked(
        self,
        *,
        kind: str,
        symbol: str,
        ratio: float | None,
        cash_per_share: float | None,
        new_symbol: str | None,
        at_ns: int | None,
    ) -> dict[str, Any]:
        ts = self._clock(at_ns)
        cleaned = _symbol(symbol)
        account = self.broker.account()
        plan = plan_action(
            kind=kind,
            symbol=cleaned,
            cash=float(account.cash),
            positions={str(key): float(value) for key, value in account.positions.items()},
            marks=dict(self.marks),
            working=self._symbol_working(cleaned),
            ratio=ratio,
            cash_per_share=cash_per_share,
            new_symbol=new_symbol,
            qty_cap=float(self.pretrade.limits.max_position_quantity),
        )
        if plan.apply:
            self._commit_action(plan, ts)
        elif plan.halt:
            self._halt_for_action(plan.symbol, ts)
        record = _action_record(kind, plan, ts)
        self._actions.append(record)
        if plan.apply:
            self._sync_account(ts)
            self._enforce_kill(ts)
        self._persist()
        self._audit("corporate_action", record)
        return _action_reply(record)

    def _commit_action(self, plan: ActionPlan, ts: int) -> None:
        self._restate_broker(plan.cash, plan.positions)
        self._replace_marks(plan.marks)
        self._model_settled += float(plan.cash_delta)
        if plan.new_symbol != plan.symbol:
            self._rename_book(plan.symbol, plan.new_symbol, plan.qty_factor, ts)
        else:
            self._scale_lots(plan.symbol, plan.qty_factor)
        self._clear_action_halt(plan, ts)
        renamed = None if plan.new_symbol == plan.symbol else plan.new_symbol
        self.ledger.apply_factors(
            symbol=plan.symbol,
            qty_factor=plan.qty_factor,
            cash_delta=plan.cash_delta,
            new_symbol=renamed,
        )

    def _halt_for_action(self, symbol: str, ts: int) -> None:
        self._halted[symbol] = True
        self._action_halt.add(symbol)
        self._reinstall_controls(symbol, ts)

    def _clear_action_halt(self, plan: ActionPlan, ts: int) -> None:
        symbol = plan.new_symbol
        if plan.symbol not in self._action_halt and symbol not in self._action_halt:
            return
        self._action_halt.discard(plan.symbol)
        self._action_halt.discard(symbol)
        self._halted[symbol] = False
        self._reinstall_controls(symbol, ts)

    def _rename_book(self, old: str, new: str, factor: float, ts: int) -> None:
        broker_name = self._symbol_map.pop(old, old)
        self._symbol_map[new] = new if broker_name == old else broker_name
        feed = self._feeds.pop(old, None)
        if feed is not None:
            self._feeds[new] = feed
        for row in self.orders:
            if str(row["symbol"]) == old:
                row["symbol"] = new
        self._move_engine_symbol(old, new, factor, ts)
        if old in self._halted and new not in self._halted:
            self._halted[new] = self._halted.pop(old)
        else:
            self._halted.pop(old, None)
        if old in self._locate:
            self._locate[new] = self._locate.pop(old)
        move_entry(self._quote_ok, old, new)
        move_entry(self._bands, old, new)

    def _move_engine_symbol(self, old: str, new: str, factor: float, ts: int) -> None:
        sid = self._sids.get(old)
        if sid is None:
            return
        sym = self.engine.book.syms[sid]
        mark = float(self.marks.get(new, 0.0))
        lots = [
            (float(sym.lq[index]), int(sym.ls[index]), int(sym.lu[index]))
            for index in range(int(sym.nlot))
        ]
        self.engine.set_symbol(
            sid, pos=0.0, ref_px=mark if mark > 0.0 else 1.0, ref_ts_ns=ts, bid=mark
        )
        for index in range(int(sym.nlot)):
            sym.lq[index] = 0.0
        sym.nlot = 0
        if mark <= 0.0:
            return
        qty = float(self.broker.account().positions.get(new, 0.0))
        new_sid = self._install(new, qty, mark, ts, seed_lots=False)
        for lot_qty, settles, funded in lots:
            scaled = lot_qty * factor
            if scaled > 0.0:
                self.engine.restore_lot(
                    new_sid, qty=scaled, settles_ns=settles, unsettled_funded=bool(funded)
                )

    def _scale_lots(self, symbol: str, factor: float) -> None:
        if factor == 1.0:
            return
        sid = self._sids.get(symbol)
        if sid is None:
            return
        sym = self.engine.book.syms[sid]
        for index in range(int(sym.nlot)):
            sym.lq[index] = float(sym.lq[index]) * factor

    def _restate_broker(self, cash: float, positions: dict[str, float]) -> None:
        broker: Any = self.broker
        if not hasattr(broker, "cash") or not hasattr(broker, "positions"):
            raise LiveGateError("broker cannot restate a corporate action offline")
        broker.cash = float(cash)
        broker.positions = {str(key): float(value) for key, value in positions.items()}

    def _replace_marks(self, marks: dict[str, float]) -> None:
        broker_marks = getattr(self.broker, "marks", None)
        self.marks = {str(key): float(value) for key, value in marks.items()}
        if isinstance(broker_marks, dict):
            for symbol in list(broker_marks):
                if symbol not in self.marks:
                    broker_marks.pop(symbol, None)
        for symbol, price in self.marks.items():
            self.broker.set_mark(symbol, price)

    def _market_block(self, symbol: str, ts: int, qty: float, price: float) -> tuple[str, ...]:
        if self._nyse_closed(ts):
            return ("market_closed",)
        if self._quote_blocks(symbol):
            return ("invalid_quote",)
        if self._quote_is_stale(symbol, ts) or self._feed_stale(symbol):
            return ("stale_data",)
        reasons: list[str] = []
        jumped = jump_problem(self.marks.get(symbol), price)
        if jumped:
            reasons.append(jumped)
        if outside_band(price, self._bands.get(symbol)):
            reasons.append("luld")
        ref = snapshot_problem(self._reference, symbol, ts)
        if ref:
            reasons.append(ref)
        elif self._reference is not None:
            reasons.extend(
                size_problems(
                    qty,
                    price,
                    self._reference.lot_size[symbol],
                    self._reference.tick_size[symbol],
                )
            )
        return tuple(reasons)

    def _feed_stale(self, symbol: str) -> bool:
        row = self._feeds.get(symbol)
        return row is not None and bool(row[1])

    def _nyse_closed(self, ts: int) -> bool:
        clock = self.pretrade.session
        if clock.timezone != "America/New_York" or clock.weekends_open:
            return False
        return nyse_rth_closed(ts)

    def _quote_blocks(self, symbol: str) -> bool:
        """A rejected quote with no stored mark must not authorize the next order."""
        return self._quote_ok.get(symbol) is False and symbol not in self.marks

    def _store_quote(self, symbol: str, decision: Any, ts: int) -> tuple[bool, bool, list[str]]:
        had_mark = symbol in self.marks
        mark = decision.mark
        accept = bool(decision.accept and mark is not None)
        reasons = list(decision.reasons)
        if accept and outside_band(float(mark), self._bands.get(symbol)):
            accept = False
            reasons.append("luld")
        self._feeds[symbol] = (decision.seq, decision.stale)
        if accept and mark is not None:
            price = float(mark)
            self._quote_ok[symbol] = True
            self.marks[symbol] = price
            self.broker.set_mark(symbol, price)
            self._sync_account(ts)
            self._enforce_kill(ts)
        elif not had_mark:
            self._quote_ok[symbol] = False
        return accept, bool(decision.stale), reasons

    def _symbol_working(self, symbol: str) -> bool:
        return any(
            row["status"] in _WORKING and str(row["symbol"]) == symbol for row in self.orders
        )

    def _bind_identity(self, symbol: str) -> None:
        if symbol in self._symbol_map:
            return
        bind_map(self._symbol_map, symbol, symbol)

    def _drop_unaccepted_mark(self, symbol: str, prior: float | None) -> None:
        """A denied order must not leave the first print behind as a reference."""
        if prior is not None:
            return
        self.marks.pop(symbol, None)
        broker_marks = getattr(self.broker, "marks", None)
        if isinstance(broker_marks, dict):
            broker_marks.pop(symbol, None)

    def _translate_fill(self, fill: BrokerFill) -> BrokerFill:
        internal = self._known_fill_symbol(fill.symbol)
        if internal == fill.symbol:
            return fill
        return replace(fill, symbol=internal)

    def _known_fill_symbol(self, symbol: str) -> str:
        try:
            return require_internal(self._symbol_map, symbol)
        except MarketDataError:
            cleaned = symbol.strip().upper()
            if cleaned in self._symbol_map:
                return cleaned
            raise LiveGateError(f"unknown broker symbol {cleaned}") from None

    def _require_reference(self, snapshot: ReferenceSnapshot) -> None:
        if snapshot.version < 1 or snapshot.max_age_ns < 0 or snapshot.as_of_ns < 0:
            raise LiveGateError("bad_reference_version")
        if set(snapshot.lot_size) != set(snapshot.tick_size):
            raise LiveGateError("bad_reference")
        for name in snapshot.lot_size:
            if price_problem(snapshot.lot_size[name]) or price_problem(snapshot.tick_size[name]):
                raise LiveGateError("bad_reference")

    def _quote_is_stale(self, symbol: str, ts: int) -> bool:
        """True when the stored quote is older than ``ts`` allows. Does not refresh it.

        A timestamp inside the current window, including a loopback ``--at``
        earlier than the wall clock, still refreshes on submit. Only an aged
        quote is refused, so submit cannot launder it by rewriting the mark.
        """
        book = self.engine.book
        if int(book.mark_ts) != 0 and ts > int(book.mark_deadline):
            return True
        return self._symbol_quote_is_stale(symbol, ts)

    def _symbol_quote_is_stale(self, symbol: str, ts: int) -> bool:
        sid = self._sids.get(symbol)
        if sid is None:
            return False
        sym = self.engine.book.syms[sid]
        if not sym.ref_ok:
            return False
        if ts < int(sym.ref_ts):
            return True
        return ts > int(sym.deadline)

    def _quote_moves_backward(self, symbol: str, ts: int) -> bool:
        sid = self._sids.get(symbol)
        if sid is None:
            return False
        sym = self.engine.book.syms[sid]
        if not sym.ref_ok:
            return False
        return ts < int(sym.ref_ts)

    def _engine_from_fill(self, fill: BrokerFill, ts: int) -> None:
        account = self.broker.account()
        pos_after = float(account.positions.get(fill.symbol, 0.0))
        signed = fill.quantity if fill.side == "buy" else -fill.quantity
        mark = self.marks.get(fill.symbol, fill.price)
        sid = self._install(fill.symbol, pos_after, mark, ts, seed_lots=False)
        session_id = self._session_id(ts)
        self.engine.set_settled_cash(self._model_settled)
        self.engine.note_fill(
            symbol_id=sid,
            side=1 if fill.side == "buy" else -1,
            qty=fill.quantity,
            px=fill.price,
            fee=0.0,
            ts_ns=ts,
            session_id=session_id,
            pos_before=pos_after - signed,
        )
        self._model_settled = float(self.engine.book.settled)
        self.engine.set_symbol(
            sid,
            pos=pos_after,
            ref_px=mark,
            ref_ts_ns=ts,
            bid=mark,
            halted=self._halted.get(fill.symbol, False),
            locate_ok=self._locate.get(fill.symbol, True),
        )

    def _sync_account(self, ts: int, *, reset_session: bool = False) -> None:
        account = self.broker.account()
        if not math.isfinite(account.cash) or account.cash < 0.0:
            raise LiveGateError("broker cash is not usable")
        if not math.isfinite(account.equity) or account.equity <= 0.0:
            raise LiveGateError("broker equity is not usable")
        self._absorb_broker_marks(account)
        reserved_qty, reserved_cash = self._working_exposure()
        self._require_marks(account.positions, reserved_qty)
        self.engine.update_account(
            nav=account.equity,
            mark_ts_ns=ts,
            session_start_nav=account.equity if reset_session else None,
            peak_nav=account.equity if reset_session else None,
        )
        available = self._model_settled - reserved_cash
        self.engine.set_settled_cash(available if available > 0.0 else 0.0)
        symbols = set(account.positions) | set(reserved_qty) | set(self.marks)
        for symbol in sorted(symbols):
            mark = self.marks.get(symbol)
            if mark is None or mark <= 0.0:
                continue
            qty = float(account.positions.get(symbol, 0.0)) + float(reserved_qty.get(symbol, 0.0))
            self._install(symbol, qty, mark, ts)
        self._latch_if_loss(ts)

    def _absorb_broker_marks(self, account: BrokerAccount) -> None:
        for symbol, price in account.marks.items():
            if price > 0.0 and symbol not in self.marks:
                self.marks[symbol] = float(price)
                self.broker.set_mark(symbol, float(price))

    def _require_marks(self, positions: dict[str, float], reserved: dict[str, float]) -> None:
        symbols = set(positions) | set(reserved)
        for symbol in sorted(symbols):
            held = float(positions.get(symbol, 0.0))
            pending = float(reserved.get(symbol, 0.0))
            if abs(held) <= 1e-9 and abs(pending) <= 1e-9:
                continue
            mark = self.marks.get(symbol)
            if mark is None or mark <= 0.0:
                raise LiveGateError(
                    f"broker position {symbol} has no mark; submission is refused until it is marked"
                )

    def _working_exposure(self) -> tuple[dict[str, float], float]:
        """Unfilled working quantity and buy notional still reserved at the broker."""
        qty: dict[str, float] = {}
        cash = 0.0
        for row in self.orders:
            if row["status"] not in _WORKING:
                continue
            remaining = float(row["quantity"]) - float(row["filled_quantity"])
            if remaining <= 1e-9:
                continue
            symbol = str(row["symbol"])
            signed = remaining if row["side"] == "buy" else -remaining
            qty[symbol] = qty.get(symbol, 0.0) + signed
            if row["side"] == "buy":
                cash += remaining * float(row["limit_price"])
        return qty, cash

    def _project_fills_onto_orders(self) -> None:
        """Copy reconciled fills onto the order row so cancel sees a filled order."""
        grouped: dict[str, list[LedgerFill]] = {}
        for fill in self.ledger.fills():
            if not self.ledger.was_applied(fill.fill_id):
                continue
            grouped.setdefault(fill.client_order_id, []).append(fill)
        for row in self.orders:
            fills = grouped.get(str(row["client_order_id"]))
            if not fills:
                continue
            qty = sum(fill.quantity for fill in fills)
            cost = sum(fill.quantity * fill.price for fill in fills)
            row["filled_quantity"] = qty
            if qty > 0.0:
                row["fill_price"] = cost / qty
            if qty + 1e-9 >= float(row["quantity"]):
                row["status"] = "filled"
                row["expects_fill"] = True
            elif qty > 1e-9 and row["status"] not in _TERMINAL:
                row["status"] = "partially_filled"
                row["expects_fill"] = True
            client = str(row["client_order_id"])
            if client in self.ledger._expects:
                self.ledger.set_expects_fill(client, bool(row["expects_fill"]))

    def _install(
        self, symbol: str, qty: float, mark: float, ts: int, *, seed_lots: bool = True
    ) -> int:
        sid = self.engine.ensure_symbol(symbol)
        if sid < 0:
            raise LiveGateError("symbol capacity is full")
        self.engine.set_symbol(
            sid,
            pos=qty,
            ref_px=mark,
            ref_ts_ns=ts,
            bid=mark,
            halted=self._halted.get(symbol, False),
            locate_ok=self._locate.get(symbol, True),
        )
        if seed_lots and qty > 0.0 and self.engine.book.syms[sid].nlot == 0:
            self.engine.restore_lot(sid, qty=qty, settles_ns=ts, unsettled_funded=False)
        self._sids[symbol] = sid
        return sid

    def _view(self, row: dict[str, Any], price: float, ts: int, kind: int) -> OrderView:
        symbol = str(row["symbol"])
        if symbol not in self._sids:
            mark = self.marks.get(symbol, price)
            self._install(symbol, 0.0, mark, ts)
        return OrderView(
            symbol_id=self._sids[symbol],
            side=1 if row["side"] == "buy" else -1,
            qty=float(row["quantity"]),
            px=float(price),
            ts_ns=ts,
            session_id=self._session_id(ts),
            is_limit=1,
            kind=kind,
        )

    def _precision_denial(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        client_order_id: str,
        actor: str,
    ) -> dict[str, Any] | None:
        reasons = precision_reasons(
            quantity,
            limit_price,
            float(self.pretrade.limits.qty_tick),
            float(self.pretrade.limits.price_tick),
        )
        if not reasons:
            return None
        self._audit(
            "deny", {"actor": actor, "client_order_id": client_order_id, "reasons": list(reasons)}
        )
        self._persist()
        return _denied(
            reasons=reasons,
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            client_order_id=client_order_id,
            venue=self.venue,
        )

    def _set_controls_locked(
        self,
        *,
        symbol: str,
        halted: bool | None,
        locate_ok: bool | None,
        at_ns: int | None,
    ) -> dict[str, Any]:
        cleaned = _symbol(symbol)
        if halted is not None:
            self._halted[cleaned] = bool(halted)
        if locate_ok is not None:
            self._locate[cleaned] = bool(locate_ok)
        self._reinstall_controls(cleaned, self._clock(at_ns))
        self._persist()
        self._audit(
            "symbol_controls",
            {
                "symbol": cleaned,
                "halted": self._halted.get(cleaned, False),
                "locate_ok": self._locate.get(cleaned, True),
            },
        )
        return {
            "ok": True,
            "symbol": cleaned,
            "halted": self._halted.get(cleaned, False),
            "locate_ok": self._locate.get(cleaned, True),
            "live_pnl_claim": False,
            "research_evidence": False,
        }

    def _reinstall_controls(self, symbol: str, ts: int) -> None:
        mark = self.marks.get(symbol)
        if mark is None or mark <= 0.0:
            return
        account = self.broker.account()
        reserved, _cash = self._working_exposure()
        qty = float(account.positions.get(symbol, 0.0)) + float(reserved.get(symbol, 0.0))
        self._install(symbol, qty, mark, ts)

    def _update_limits_locked(
        self, limits: LimitConfig, *, actor: str, reauthorize: bool
    ) -> dict[str, Any]:
        structural = structural_changes(self.pretrade.limits, limits)
        if structural:
            self._audit("limits_refused", {"actor": actor, "structural": list(structural)})
            return _limit_reply(applied=False, structural=structural)
        looser = loosened_names(self.pretrade.limits, limits)
        if looser and not reauthorize:
            self._audit("limits_refused", {"actor": actor, "loosened": list(looser)})
            return _limit_reply(applied=False, loosened=looser)
        self._apply_limit_config(limits)
        ts = self._clock(None)
        self._latch_if_loss(ts)
        self._enforce_kill(ts)
        self._persist()
        self._audit(
            "limits",
            {"actor": actor, "reauthorized": reauthorize, "loosened": list(looser)},
        )
        return _limit_reply(applied=True, loosened=looser, reauthorized=reauthorize)

    def _apply_limit_config(self, limits: LimitConfig) -> None:
        install_ceilings(self.engine, limits)
        self.pretrade = self.pretrade.model_copy(update={"limits": limits})
        digest, signature = sign_config(self.pretrade, _hmac_for_engine(self.state_dir))
        self.engine.config_sha256 = digest
        self.engine.config_hmac_sha256 = signature
        self._limit_override = limits

    def _restore_limits(self) -> None:
        if self._limit_override is not None:
            self._apply_limit_config(self._limit_override)

    def _latch_if_loss(self, ts: int) -> None:
        if self.kill_state["latched"]:
            return
        reason = loss_reason(
            float(self.engine.book.nav),
            float(self.engine.book.daily_floor),
            float(self.engine.book.dd_floor),
        )
        if not reason:
            return
        self.engine.trip("risk", reason, ts)
        self.kill_state = {"latched": True, "actor": "risk", "reason": reason}

    def _enforce_kill(self, ts: int) -> list[str]:
        if not self.kill_state["latched"]:
            return []
        # A position break stops new orders. It does not cancel a partial fill.
        if (
            self.kill_state.get("actor") == "reconcile"
            and self.kill_state.get("state") == "HALT_NEW_ORDERS"
        ):
            return []
        return self._cancel_working(ts)

    def _cancel_working(self, ts: int) -> list[str]:
        """Cancel every working order. A broker reject leaves that order working."""
        sent: list[str] = []
        for row in list(self.orders):
            if row["status"] not in _WORKING:
                continue
            try:
                ack = self.broker.cancel(str(row["client_order_id"]), str(row["broker_order_id"]))
            except BrokerRejected as exc:
                row["reason"] = str(exc)
                continue
            self._apply_ack(row, ack, ts)
            if row["status"] not in _WORKING:
                sent.append(str(row["client_order_id"]))
        return sent

    def _working_ids(self) -> list[str]:
        return [str(row["client_order_id"]) for row in self.orders if row["status"] in _WORKING]

    def _working_has_broker_id(self) -> bool:
        return any(row["status"] in _WORKING and str(row["broker_order_id"]) for row in self.orders)

    def _controls_payload(self) -> dict[str, dict[str, bool]]:
        symbols = set(self._halted) | set(self._locate)
        return {
            symbol: {
                "halted": self._halted.get(symbol, False),
                "locate_ok": self._locate.get(symbol, True),
            }
            for symbol in sorted(symbols)
        }

    def _read_controls(self, stored: dict[str, Any]) -> None:
        self._halted = {}
        self._locate = {}
        raw = stored.get("symbol_controls") or {}
        if not isinstance(raw, dict):
            raise LiveGateError("stored symbol controls are malformed")
        for symbol, item in raw.items():
            if not isinstance(item, dict):
                raise LiveGateError("stored symbol control is malformed")
            name = _symbol(str(symbol))
            self._halted[name] = bool(item.get("halted"))
            if "locate_ok" in item:
                self._locate[name] = bool(item["locate_ok"])

    def _read_market(self, stored: dict[str, Any]) -> None:
        self._actions = _action_list(stored.get("corporate_actions"))
        halted = stored.get("action_halt") or []
        if not isinstance(halted, list):
            raise LiveGateError("action halt set is malformed")
        self._action_halt = {str(item) for item in halted}
        self._feeds = _feed_map(stored.get("feeds"))
        self._load_clock(stored)
        self._reference = _reference_from_state(stored.get("reference"))
        if "symbol_map" not in stored:
            self._symbol_map = {}
            self._seed_identity(stored.get("fills"))
            return
        raw_map = stored.get("symbol_map")
        if not isinstance(raw_map, dict):
            raise LiveGateError("symbol map is malformed")
        self._symbol_map = {str(key): str(value) for key, value in raw_map.items()}
        if self._symbol_map and not mapping_is_bijective(self._symbol_map):
            raise LiveGateError("symbol map is not bijective")

    def _load_clock(self, stored: dict[str, Any]) -> None:
        try:
            self._quote_ok = flag_map(stored.get("quote_ok"))
            self._bands = band_map(stored.get("price_bands"))
        except ClockError as exc:
            raise LiveGateError(str(exc)) from exc

    def _seed_identity(self, fills: object) -> None:
        names = set(self.marks)
        names.update(str(row["symbol"]) for row in self.orders)
        if isinstance(fills, list):
            for item in fills:
                if isinstance(item, dict) and item.get("symbol"):
                    names.add(str(item["symbol"]))
        for name in sorted(names):
            if name not in self._symbol_map:
                bind_map(self._symbol_map, name, name)

    def _apply_stored_factors(self) -> None:
        for action in self._actions:
            if not action.get("apply"):
                continue
            symbol = str(action["symbol"])
            new_symbol = str(action["new_symbol"])
            renamed = None if new_symbol == symbol else new_symbol
            self.ledger.apply_factors(
                symbol=symbol,
                qty_factor=float(action["qty_factor"]),
                cash_delta=float(action["cash_delta"]),
                new_symbol=renamed,
            )

    def _reapply_actions_to_engine(self, ts: int) -> None:
        for action in self._actions:
            if not action.get("apply"):
                continue
            symbol = str(action["symbol"])
            new_symbol = str(action["new_symbol"])
            factor = float(action["qty_factor"])
            self._model_settled += float(action["cash_delta"])
            self._restamp_engine(symbol, new_symbol, factor, ts)

    def _restamp_engine(self, symbol: str, new_symbol: str, factor: float, ts: int) -> None:
        sid = self._sids.get(symbol)
        if sid is None:
            return
        sym = self.engine.book.syms[sid]
        mark = float(self.marks.get(new_symbol, self.marks.get(symbol, 0.0)))
        if mark <= 0.0:
            return
        if new_symbol != symbol:
            self._move_engine_symbol(symbol, new_symbol, factor, ts)
            return
        qty = float(sym.pos) * factor
        self.engine.set_symbol(
            sid,
            pos=qty,
            ref_px=mark,
            ref_ts_ns=ts,
            bid=mark,
            halted=self._halted.get(symbol, False),
            locate_ok=self._locate.get(symbol, True),
        )
        if factor != 1.0:
            for index in range(int(sym.nlot)):
                sym.lq[index] = float(sym.lq[index]) * factor

    def _read_limit_override(self, stored: dict[str, Any]) -> None:
        payload = stored.get("limit_override")
        if payload is None:
            self._limit_override = None
            return
        if not isinstance(payload, dict):
            raise LiveGateError("stored limit override is malformed")
        self._limit_override = LimitConfig.model_validate(payload)

    def _session_id(self, ts: int) -> int:
        # shadow imports the simulated broker. Keep that off the CLI import path.
        from quant_fund.pretrade.shadow import business_session_id

        zone = ZoneInfo(self.pretrade.session.timezone)
        local = datetime.fromtimestamp(ts / 1_000_000_000, zone)
        return business_session_id(local.date())

    def _capture_baseline(self) -> None:
        if self.venue == "loopback" or self._baseline_from_broker:
            return
        if self.orders or self.ledger.fills():
            return
        account = self.broker.account()
        self.ledger = LiveLedger(
            baseline_cash=account.cash,
            baseline_positions=dict(account.positions),
            cash_tolerance=self.config.cash_tolerance,
            position_tolerance=self.config.position_tolerance,
        )
        self._baseline_from_broker = True

    def _recover_inflight(self) -> bool:
        """Recover inflight orders. False when none were inflight or a lookup failed."""
        ts = self._now()
        changed = False
        failed = False
        saw = False
        for row in self.orders:
            if row["status"] != "inflight":
                continue
            saw = True
            ack = self._safe_recover(str(row["client_order_id"]))
            if ack is None:
                failed = True
                continue
            changed = True
            self._apply_ack(row, ack, ts)
        if changed:
            self._persist()
        return saw and not failed

    def _require_trading(self) -> None:
        if self._arm is None:
            raise LiveGateError("session is not armed; run live arm")
        if not _credentials_present(self.broker) and self.venue != "loopback":
            raise LiveGateError("broker credentials are not set")

    def _clock(self, at_ns: int | None) -> int:
        if at_ns is None:
            return int(self._now())
        if self.venue == "alpaca-live":
            raise LiveGateError(
                "--at is refused for alpaca-live; the session clock is the wall clock"
            )
        if at_ns < 0:
            raise LiveGateError("timestamp must be non-negative")
        return int(at_ns)

    def _find(self, client_order_id: str) -> dict[str, Any] | None:
        for row in self.orders:
            if row["client_order_id"] == client_order_id:
                return row
        return None

    def _opening_balances(self) -> tuple[float, float]:
        if self.venue != "loopback":
            cash = float(self.ledger.computed_cash)
            nav = cash if cash > 0.0 else float(self.config.starting_cash)
            return nav, cash
        account = self.broker.account()
        nav = account.equity if account.equity > 0.0 else account.cash
        if nav <= 0.0:
            nav = float(self.config.starting_cash)
        return nav, account.cash

    def _load_state(self, stored: dict[str, Any] | None) -> None:
        if stored is None:
            self._load_fresh()
            return
        self._load_stored(stored)

    def _load_fresh(self) -> None:
        self.orders = []
        self.marks = {}
        self.kill_state = {"latched": False, "actor": "", "reason": ""}
        self._halted = {}
        self._locate = {}
        self._limit_override = None
        self._symbol_map = {}
        self._feeds = {}
        self._reference = None
        self._actions = []
        self._action_halt = set()
        self._quote_ok = {}
        self._bands = {}
        self._pretrade_blob = None
        self._baseline_from_broker = self.venue == "loopback"
        self.ledger = self._new_ledger(float(self.config.starting_cash), {})
        if self.venue == "loopback" and not isinstance(self.broker, LoopbackBroker):
            raise LiveGateError("loopback venue requires the loopback broker")
        if isinstance(self.broker, LoopbackBroker):
            self.broker.cash = float(self.config.starting_cash)

    def _load_stored(self, stored: dict[str, Any]) -> None:
        _require_state_identity(stored, self.venue)
        orders, fills, marks, kill = _state_collections(stored)
        self.orders = [_order_row(item) for item in orders]
        self.marks = {
            str(symbol): _positive(float(price), "stored mark") for symbol, price in marks.items()
        }
        self.kill_state = _kill_from_state(kill)
        self._read_controls(stored)
        self._read_limit_override(stored)
        self._read_market(stored)
        self._pretrade_blob = stored.get("pretrade")
        baseline_positions = stored.get("baseline_positions") or {}
        if not isinstance(baseline_positions, dict):
            raise LiveGateError("baseline positions are malformed")
        self._baseline_from_broker = bool(
            stored.get("baseline_from_broker", self.venue == "loopback")
        )
        self.ledger = self._new_ledger(
            float(stored.get("baseline_cash", self.config.starting_cash)),
            {str(key): float(value) for key, value in baseline_positions.items()},
        )
        self._replay(fills)
        self._apply_stored_factors()
        if isinstance(self.broker, LoopbackBroker):
            self.broker.cash = self.ledger.computed_cash
            self.broker.positions = self.ledger.computed_positions

    def _new_ledger(self, cash: float, positions: dict[str, float]) -> LiveLedger:
        return LiveLedger(
            baseline_cash=cash,
            baseline_positions=positions,
            cash_tolerance=self.config.cash_tolerance,
            position_tolerance=self.config.position_tolerance,
        )

    def _replay(self, fills: list[Any]) -> None:
        for row in self.orders:
            self.ledger.note_order(
                str(row["client_order_id"]),
                expects_fill=bool(row["expects_fill"]),
                quantity=float(row["quantity"]),
            )
        for item in fills:
            if not isinstance(item, dict):
                raise LiveGateError("stored fill is malformed")
            self.ledger.record_fill(ledger_fill_from_dict(item))

    def _persist(self) -> None:
        payload: dict[str, Any] = {
            "schema": _STATE_SCHEMA,
            "venue": self.venue,
            "baseline_cash": self.ledger.baseline_cash,
            "baseline_positions": self.ledger.baseline_positions,
            "baseline_from_broker": self._baseline_from_broker,
            "marks": dict(self.marks),
            "orders": list(self.orders),
            "fills": [fill.as_dict() for fill in self.ledger.fills()],
            "kill": dict(self.kill_state),
            "symbol_controls": self._controls_payload(),
            "limit_override": _limit_payload(self._limit_override),
            "symbol_map": dict(self._symbol_map),
            "feeds": {
                symbol: {"seq": seq, "stale": stale} for symbol, (seq, stale) in self._feeds.items()
            },
            "quote_ok": dict(self._quote_ok),
            "price_bands": {
                symbol: {"lower": lower, "upper": upper}
                for symbol, (lower, upper) in self._bands.items()
            },
            "reference": None if self._reference is None else self._reference.as_dict(),
            "corporate_actions": list(self._actions),
            "action_halt": sorted(self._action_halt),
            "pretrade": export_pretrade(self.engine),
            "live_pnl_claim": False,
            "research_evidence": False,
        }
        _atomic_json(self.state_path, payload)

    def _audit(self, event: str, fields: dict[str, Any]) -> None:
        body: dict[str, Any] = {
            "event": event,
            "prev_hash": self.audit_head,
            "live_pnl_claim": False,
            "research_evidence": False,
        }
        body.update(fields)
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        line = dict(body)
        line["event_hash"] = digest
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_line = (json.dumps(line, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(str(self.audit_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, encoded_line)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(self.audit_path, 0o600)
        self.audit_head = digest

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with self._thread_lock, self._file_lock():
            yield

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.state_dir, 0o700)
        path = self.state_dir / "lock"
        handle = path.open("a+b")
        try:
            if path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            os.chmod(path, 0o600)
            _acquire(handle)
            try:
                token = self._file_token()
                if token != self._disk_token:
                    self._reload_locked()
                yield
                self._disk_token = self._file_token()
            finally:
                _release(handle)
        finally:
            handle.close()

    def _file_token(self) -> str:
        digest = hashlib.sha256()
        for path in (self.arm_path, self.state_path, self.audit_path):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes() if path.exists() else b"-")
        return digest.hexdigest()

    def _reload_locked(self) -> None:
        """Read the desk from disk. The caller already holds the process lock."""
        if self.audit_path.exists():
            os.chmod(self.audit_path, 0o600)
        try:
            self._arm = read_arm(
                self.arm_path, venue=self.venue, pretrade_sha256=self.pretrade_sha256
            )
            stored = _read_object(self.state_path)
            self.audit_head = _verify_audit(self.audit_path)
            self._load_state(stored)
        except LiveGateError as exc:
            self._latch_restore(str(exc))
            raise
        self._build_engine()
        self._restore_limits()
        for symbol, price in self.marks.items():
            self.broker.set_mark(symbol, price)
        if self.venue == "loopback":
            self._sync_account(self._now())
            self._enforce_kill(self._now())
        if self.kill_state["latched"]:
            self.engine.trip(
                str(self.kill_state["actor"]), str(self.kill_state["reason"]), self._now()
            )
        was_pending = self._open_needs_reconcile()
        recovered = self._recover_inflight()
        self._reconcile_after_reload(was_pending, recovered)

    def _reconcile_after_reload(self, was_pending: bool, recovered: bool) -> None:
        """Pull the fills page after a restart once the broker answers, or while halted."""
        if self._arm is None:
            return
        if self.kill_state.get("latched") or (was_pending and recovered):
            self._reconcile_locked(None, ts=self._adoption_clock())

    def _build_engine(self) -> None:
        """New engine, then the persisted rings and a fill replay for settlement."""
        cash = float(self.ledger.baseline_cash)
        if cash <= 0.0:
            cash = float(self.config.starting_cash)
        self._model_settled = cash
        self.engine = PretradeEngine(
            self.pretrade,
            hmac_key=_hmac_for_engine(self.state_dir),
            initial_nav=cash,
            initial_cash=cash,
            ts_ns=self._now(),
        )
        self._model_settled = float(self.engine.book.settled)
        self._sids = {}
        self._engine_noted = set()
        if self._pretrade_blob is not None:
            restore_pretrade(self.engine, self._pretrade_blob)
        self._replay_fills_into_engine(self._now())
        self._reapply_actions_to_engine(self._now())

    def _replay_fills_into_engine(self, ts: int) -> None:
        positions: dict[str, float] = {}
        for fill in self.ledger.fills():
            if not self.ledger.was_applied(fill.fill_id):
                continue
            signed = fill.quantity if fill.side == "buy" else -fill.quantity
            before = positions.get(fill.symbol, 0.0)
            after = before + signed
            mark = self.marks.get(fill.symbol, fill.price)
            sid = self._install(fill.symbol, after, mark, ts, seed_lots=False)
            self.engine.set_settled_cash(self._model_settled)
            self.engine.note_fill(
                symbol_id=sid,
                side=1 if fill.side == "buy" else -1,
                qty=fill.quantity,
                px=fill.price,
                fee=0.0,
                ts_ns=fill.fill_time_ns if fill.fill_time_ns > 0 else ts,
                session_id=self._session_id(fill.fill_time_ns if fill.fill_time_ns > 0 else ts),
                pos_before=before,
            )
            self._model_settled = float(self.engine.book.settled)
            self._engine_noted.add(fill.client_order_id)
            positions[fill.symbol] = after


def open_session(
    config: LiveConfig,
    config_path: str | Path,
    *,
    state_dir: str | Path | None = None,
    now_ns: Callable[[], int] | None = None,
    broker: Broker | None = None,
    transport: Any = None,
) -> LiveSession:
    """Open or create the session directory for ``config``."""
    root = Path(state_dir) if state_dir is not None else Path(config.state_dir)
    root.mkdir(parents=True, exist_ok=True)
    key = load_hmac_key(root / "hmac.key")
    pretrade, digest, _signature = load_gateway_pretrade(config, config_path, hmac_key=key)
    clock = now_ns or _wall_ns
    if broker is None:
        broker = _default_broker(config, transport=transport)
    return LiveSession(
        config,
        state_dir=root,
        pretrade=pretrade,
        pretrade_sha256=digest,
        broker=broker,
        now_ns=clock,
    )


def _default_broker(config: LiveConfig, *, transport: Any) -> Broker:
    if config.venue == "loopback":
        return LoopbackBroker(cash=float(config.starting_cash))
    return AlpacaBroker(config.venue, alpaca_credentials(), transport=transport)


def _hmac_for_engine(state_dir: Path) -> bytes:
    return (state_dir / "hmac.key").read_bytes()


def _wall_ns() -> int:
    return time.time_ns()


def _credentials_present(broker: Broker) -> bool:
    if isinstance(broker, AlpacaBroker):
        return broker.credentials_present
    return True


def _line_is_torn(raw_line: str) -> bool:
    try:
        json.loads(raw_line)
    except json.JSONDecodeError:
        return True
    return False


def _verify_audit(path: Path) -> str:
    head = zero_audit_head()
    if not path.exists():
        return head
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines and _line_is_torn(lines[-1]):
        lines = lines[:-1]
        _rewrite_lines(path, lines)
    for raw_line in lines:
        head = _audit_head(raw_line, head)
    return head


def _rewrite_lines(path: Path, lines: list[str]) -> None:
    """Replace a journal with its intact prefix. The torn tail is not replayed."""
    payload = "".join(f"{line}\n" for line in lines).encode()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _audit_head(raw_line: str, head: str) -> str:
    try:
        parsed = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise LiveGateError("audit log line is truncated") from exc
    if not isinstance(parsed, dict):
        raise LiveGateError("audit log line is not an object")
    row = dict(parsed)
    event_hash = row.pop("event_hash", None)
    if row.get("prev_hash") != head:
        raise LiveGateError("audit log chain does not match")
    if row.get("live_pnl_claim") is not False or row.get("research_evidence") is not False:
        raise LiveGateError("audit log claims live profit or research evidence")
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != event_hash:
        raise LiveGateError("audit log line was modified")
    return digest


def _require_state_identity(stored: dict[str, Any], venue: str) -> None:
    if stored.get("schema") != _STATE_SCHEMA:
        raise LiveGateError("session state schema is not supported")
    if stored.get("venue") != venue:
        raise LiveGateError("session state venue does not match the live config")
    profit = stored.get("live_pnl_claim") is not False
    evidence = stored.get("research_evidence") is not False
    if profit or evidence:
        raise LiveGateError("session state must not claim research evidence or live profit")


def _state_collections(
    stored: dict[str, Any],
) -> tuple[list[Any], list[Any], dict[str, Any], dict[str, Any]]:
    orders = stored.get("orders")
    fills = stored.get("fills")
    marks = stored.get("marks")
    kill = stored.get("kill")
    if not isinstance(orders, list) or not isinstance(fills, list) or not isinstance(marks, dict):
        raise LiveGateError("session state is missing orders, fills, or marks")
    if not isinstance(kill, dict):
        raise LiveGateError("session state is missing the kill switch")
    return orders, fills, marks, kill


def _position_alerts(
    report: dict[str, Any],
    *,
    cash_tolerance: float,
    position_tolerance: float,
) -> list[str]:
    """Name each broker mismatch outside tolerance. No declaration means no alert."""
    drift = report.get("drift")
    if not isinstance(drift, dict) or not drift.get("declared"):
        return []
    alerts: list[str] = []
    cash_drift = drift.get("cash_drift")
    if cash_drift is not None and float(cash_drift) > cash_tolerance:
        alerts.append("cash mismatch")
    positions = drift.get("position_drift")
    if isinstance(positions, dict):
        for symbol in sorted(positions):
            if float(positions[symbol]) > position_tolerance:
                alerts.append(f"position mismatch {symbol}")
    return alerts


_KILL_STATES = frozenset({"ENABLED", "HALT_NEW_ORDERS", "CANCEL_OPEN_ORDERS", "FLATTEN_OPTIONAL"})


def _kill_from_state(kill: dict[str, Any]) -> dict[str, Any]:
    latched = bool(kill.get("latched"))
    actor = str(kill.get("actor") or "")
    reason = str(kill.get("reason") or "")
    if latched and (not actor.strip() or not reason.strip()):
        raise LiveGateError("a latched kill switch needs an actor and a reason")
    name = kill.get("state", "HALT_NEW_ORDERS" if latched else "ENABLED")
    if not isinstance(name, str) or name not in _KILL_STATES:
        raise LiveGateError(f"unknown kill switch state={name!r}")
    if name != "ENABLED":
        latched = True
    return {"latched": latched, "actor": actor, "reason": reason, "state": name}


def _order_row(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise LiveGateError("stored order is malformed")
    status = str(raw.get("status") or "")
    if status not in _ORDER_STATUSES:
        raise LiveGateError("stored order status is not supported")
    fill_price = raw.get("fill_price")
    return {
        "client_order_id": str(raw["client_order_id"]),
        "broker_order_id": str(raw.get("broker_order_id") or ""),
        "symbol": _symbol(str(raw["symbol"])),
        "side": _side(str(raw["side"])),
        "quantity": _positive(float(raw["quantity"]), "stored quantity"),
        "limit_price": _positive(float(raw["limit_price"]), "stored limit"),
        "status": status,
        "filled_quantity": float(raw.get("filled_quantity") or 0.0),
        "fill_price": None if fill_price is None else float(fill_price),
        "expects_fill": bool(raw.get("expects_fill")),
        "reason": str(raw.get("reason") or ""),
        "idempotency_key": str(raw.get("idempotency_key") or raw["client_order_id"]),
    }


def _public_order(row: dict[str, Any], *, venue: str, replayed: bool) -> dict[str, Any]:
    status = str(row["status"])
    return {
        "ok": status in {"filled", "accepted", "partially_filled", "canceled"},
        "replayed": replayed,
        "status": status,
        "client_order_id": row["client_order_id"],
        "broker_order_id": row["broker_order_id"],
        "symbol": row["symbol"],
        "side": row["side"],
        "quantity": row["quantity"],
        "limit_price": row["limit_price"],
        "filled_quantity": row["filled_quantity"],
        "fill_price": row["fill_price"],
        "reason": row["reason"],
        "reasons": [],
        "venue": venue,
        "broker_called": True,
        "idempotency_key": str(row.get("idempotency_key") or row["client_order_id"]),
        "live_pnl_claim": False,
        "research_evidence": False,
    }


def _merge_reasons(primary: tuple[str, ...] | list[str], extra: tuple[str, ...]) -> tuple[str, ...]:
    merged = list(primary)
    for reason in extra:
        if reason not in merged:
            merged.append(reason)
    return tuple(merged)


def _denied(
    *,
    reasons: tuple[str, ...],
    symbol: str,
    side: str,
    quantity: float,
    limit_price: float,
    client_order_id: str,
    venue: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "replayed": False,
        "status": "denied",
        "client_order_id": client_order_id,
        "broker_order_id": "",
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "limit_price": limit_price,
        "filled_quantity": 0.0,
        "fill_price": None,
        "reason": ",".join(reasons),
        "reasons": list(reasons),
        "venue": venue,
        "broker_called": False,
        "idempotency_key": client_order_id,
        "live_pnl_claim": False,
        "research_evidence": False,
    }


def _windows_lock(handle: Any, operation: str) -> None:
    """Byte-range lock via the Windows C runtime. ``operation`` is LK_LOCK or LK_UNLCK."""
    import importlib
    from operator import attrgetter

    runtime = importlib.import_module("msvcrt")
    handle.seek(0)
    attrgetter("locking")(runtime)(handle.fileno(), attrgetter(operation)(runtime), 1)


def _acquire(handle: Any) -> None:
    if os.name == "nt":
        _windows_lock(handle, "LK_LOCK")
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release(handle: Any) -> None:
    if os.name == "nt":
        _windows_lock(handle, "LK_UNLCK")
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_timestamp(text: str) -> int:
    """Parse an aware ISO-8601 timestamp into nanoseconds."""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise LiveGateError("--at must include a timezone offset")
    return int(parsed.timestamp() * 1_000_000_000)
