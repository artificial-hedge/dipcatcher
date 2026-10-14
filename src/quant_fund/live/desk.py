"""Operator order desk.

Research commands never import this module. Orders leave the machine only
after the authorization file is armed, the pre-trade engine allows the
order (or a human flatten is cutting an existing position), and the broker
call is bound to the paper or live host named in that file.

Nothing here is an institutional live-trading claim. ``would_promote_live``
and ``live_pnl_claim`` stay false on every record.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant_fund.live.alpaca import AccountState, AlpacaClient, PositionState
from quant_fund.live.authorization import (
    LiveAuthorization,
    assert_current,
    assert_live_confirmed,
    authorization_sha256,
    credential_flags,
    credentials_from_env,
    hmac_key_from_env,
    load_authorization,
)
from quant_fund.live.errors import BrokerRejected, BrokerUncertain, LiveRefused
from quant_fund.live.lock import desk_lock
from quant_fund.live.transport import HttpTransport, UrllibTransport
from quant_fund.pretrade.codes import KIND_ORDER, reason_names
from quant_fund.pretrade.config import PretradeConfig, load_pretrade_config
from quant_fund.pretrade.engine import Decision, OrderView, PretradeEngine
from quant_fund.schemas.errors import KillSwitchActive


def _business_session_id(day: date) -> int:
    """Weekday index. Same rule as ``quant_fund.pretrade.shadow.business_session_id``."""
    zero = day.toordinal() - 1
    return (zero // 7) * 5 + min(zero % 7, 4) + 1


_CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{8,48}$")
_REDUCE_OVERRIDE = frozenset({"kill_switch", "daily_loss", "trailing_drawdown", "manual_kill"})
_FORBIDDEN_KEYS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def honesty_stamp(*, environment: str) -> dict[str, object]:
    """Fields that keep a routing record from being read as research evidence."""
    return {
        "institutional_live_claim": False,
        "would_promote_live": False,
        "live_pnl_claim": False,
        "operator_order_routing": True,
        "environment": environment,
        "evidence_class": "broker_routing",
    }


def _wall_clock() -> datetime:
    return datetime.now(UTC)


def _ts_ns(when: datetime) -> int:
    return int(when.timestamp() * 1_000_000_000)


def _actors(actor: str, reason: str) -> tuple[str, str]:
    actor_s = actor.strip()
    reason_s = reason.strip()
    if not actor_s or not reason_s or len(actor_s) > 80 or len(reason_s) > 200:
        raise LiveRefused("actor and reason are required")
    return actor_s, reason_s


def _side(value: str) -> str:
    text = value.strip().lower()
    if text not in {"buy", "sell"}:
        raise LiveRefused("side must be buy or sell")
    return text


def _prepare_state_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if "receipts" in resolved.parts:
        raise LiveRefused("state dir must not be inside receipts")
    if resolved.exists() and not resolved.is_dir():
        raise LiveRefused("state dir is not a directory")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LiveRefused(f"{path.name} is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise LiveRefused(f"{path.name} is not a JSON object")
    return raw


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _new_client_order_id() -> str:
    return "dc" + uuid.uuid4().hex[:32]


class LiveDesk:
    """Fail-closed router. Public methods take the state lock; helpers do not."""

    def __init__(
        self,
        *,
        authorization: LiveAuthorization,
        authorization_sha256: str,
        config: PretradeConfig,
        hmac_key: bytes,
        broker: AlpacaClient,
        state_dir: str | Path,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if broker.environment != authorization.environment:
            raise LiveRefused("broker environment does not match authorization")
        if len(hmac_key) < 16:
            raise LiveRefused("hmac key must be at least 16 bytes")
        self.authorization = authorization
        self._authorization_sha256 = authorization_sha256
        self.config = config
        self._hmac_key = hmac_key
        self.broker = broker
        self.state_dir = _prepare_state_dir(Path(state_dir))
        self._clock = clock or _wall_clock
        self._tz = ZoneInfo(config.session.timezone)
        if authorization.max_order_notional > config.limits.max_order_notional:
            raise LiveRefused("authorization max_order_notional exceeds the pretrade limit")

    # -- public surface (locked) --

    def arm(self) -> dict[str, Any]:
        with self._locked():
            return self._arm_unlocked()

    def disarm(self) -> dict[str, Any]:
        with self._locked():
            return self._disarm_unlocked()

    def status(self) -> dict[str, Any]:
        with self._locked():
            return self._status_unlocked()

    def kill(self, *, actor: str, reason: str) -> dict[str, Any]:
        with self._locked():
            return self._kill_unlocked(actor=actor, reason=reason)

    def reset_kill(self, *, actor: str, reason: str) -> dict[str, Any]:
        with self._locked():
            return self._reset_kill_unlocked(actor=actor, reason=reason)

    def sync(self) -> dict[str, Any]:
        with self._locked():
            return self._sync_unlocked()

    def submit(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float | None = None,
        client_order_id: str | None = None,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        with self._locked():
            return self._submit_unlocked(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=limit_price,
                client_order_id=client_order_id,
                reduce_only=reduce_only,
            )

    def cancel(self, broker_order_id: str) -> dict[str, Any]:
        with self._locked():
            return self._cancel_unlocked(broker_order_id)

    def cancel_all(self) -> dict[str, Any]:
        with self._locked():
            return self._cancel_all_unlocked()

    def flatten(self, *, confirm: str) -> dict[str, Any]:
        with self._locked():
            return self._flatten_unlocked(confirm=confirm)

    def reconcile(self) -> dict[str, Any]:
        with self._locked():
            return self._reconcile_unlocked()

    # -- lock and clock --

    def _locked(self) -> AbstractContextManager[None]:
        return desk_lock(self.state_dir / ".desk.lock")

    def _now(self) -> datetime:
        current = self._clock()
        if current.tzinfo is None or current.utcoffset() is None:
            raise LiveRefused("clock must be timezone-aware")
        return current

    def _stamp(self) -> dict[str, object]:
        stamped = honesty_stamp(environment=self.authorization.environment)
        overlap = _FORBIDDEN_KEYS.intersection(stamped)
        if overlap:
            raise LiveRefused(f"routing record used a forbidden key: {sorted(overlap)}")
        return stamped

    # -- arm / kill / status --

    def _require_armed(self) -> None:
        raw = _read_json(self.state_dir / "armed.json")
        if raw.get("armed") is not True:
            raise LiveRefused("desk is not armed")
        if raw.get("authorization_sha256") != self._authorization_sha256:
            raise LiveRefused("authorization file changed; disarm and arm again")
        if raw.get("environment") != self.authorization.environment:
            raise LiveRefused("armed environment does not match the authorization file")
        if raw.get("account_id") != self.authorization.account_id:
            raise LiveRefused("armed account does not match the authorization file")

    def _arm_unlocked(self) -> dict[str, Any]:
        now = self._now()
        assert_current(self.authorization, now=now)
        assert_live_confirmed(self.authorization.environment)
        payload = {
            "armed": True,
            "authorization_sha256": self._authorization_sha256,
            "environment": self.authorization.environment,
            "account_id": self.authorization.account_id,
            "operator": self.authorization.operator,
            "armed_at": now.isoformat(),
        }
        self._write_checked(self.state_dir / "armed.json", payload)
        record = {"action": "arm", "armed": True, **self._stamp()}
        self._audit(record)
        return record

    def _disarm_unlocked(self) -> dict[str, Any]:
        payload = {"armed": False, "authorization_sha256": self._authorization_sha256}
        self._write_checked(self.state_dir / "armed.json", payload)
        record = {"action": "disarm", "armed": False, **self._stamp()}
        self._audit(record)
        return record

    def _status_unlocked(self) -> dict[str, Any]:
        now = self._now()
        armed = _read_json(self.state_dir / "armed.json")
        kill = _read_json(self.state_dir / "kill.json")
        return {
            "action": "status",
            "armed": armed.get("armed") is True
            and armed.get("authorization_sha256") == self._authorization_sha256,
            "kill_latched": kill.get("latched") is True,
            "kill_reason": kill.get("reason") if kill.get("latched") is True else None,
            "account_id": self.authorization.account_id,
            "operator": self.authorization.operator,
            "orders_today": self._orders_today(now),
            "max_orders_per_day": self.authorization.max_orders_per_day,
            "symbol_allowlist": list(self.authorization.symbol_allowlist),
            **self._stamp(),
        }

    def _kill_unlocked(self, *, actor: str, reason: str) -> dict[str, Any]:
        actor_s, reason_s = _actors(actor, reason)
        self._write_kill(actor_s, reason_s)
        record = {
            "action": "kill",
            "kill_latched": True,
            "actor": actor_s,
            "reason": reason_s,
            **self._stamp(),
        }
        self._audit(record)
        return record

    def _reset_kill_unlocked(self, *, actor: str, reason: str) -> dict[str, Any]:
        actor_s, reason_s = _actors(actor, reason)
        _write_json(
            self.state_dir / "kill.json",
            {"latched": False, "actor": actor_s, "reason": reason_s},
        )
        record = {
            "action": "reset_kill",
            "kill_latched": False,
            "actor": actor_s,
            "reason": reason_s,
            **self._stamp(),
        }
        self._audit(record)
        return record

    def kill_latched(self) -> bool:
        return _read_json(self.state_dir / "kill.json").get("latched") is True

    def _write_kill(self, actor: str, reason: str) -> None:
        self._write_checked(
            self.state_dir / "kill.json",
            {"latched": True, "actor": actor, "reason": reason},
        )

    # -- account sync --

    def _sync_unlocked(self) -> dict[str, Any]:
        self._require_armed()
        assert_live_confirmed(self.authorization.environment)
        now = self._now()
        account = self._checked_account()
        positions = self.broker.get_positions()
        start, peak = self._session_marks(account.equity, now)
        latched = self._maybe_latch_loss(account.equity, start, peak)
        record: dict[str, Any] = {
            "action": "sync",
            "account_status": account.status,
            "broker_equity": account.equity,
            "broker_cash": account.cash,
            "positions": [
                {"symbol": row.symbol, "qty": row.qty} for row in positions if row.qty != 0.0
            ],
            "kill_latched": latched,
            **self._stamp(),
        }
        self._audit(record)
        return record

    def _checked_account(self) -> AccountState:
        account = self.broker.get_account()
        if account.status != "ACTIVE":
            raise LiveRefused(f"broker account status is {account.status or 'missing'}")
        expected = self.authorization.account_id
        if expected not in {account.account_id, account.account_number}:
            raise LiveRefused("broker account does not match the authorization file")
        return account

    def _session_marks(self, equity: float, now: datetime) -> tuple[float, float]:
        day = now.astimezone(self._tz).date().isoformat()
        path = self.state_dir / "session.json"
        raw = _read_json(path)
        if raw.get("day") != day or "session_start_equity" not in raw or "peak_equity" not in raw:
            start = equity
            peak = equity
        else:
            start = float(raw["session_start_equity"])
            peak = max(float(raw["peak_equity"]), equity)
        self._write_checked(
            path,
            {"day": day, "session_start_equity": start, "peak_equity": peak},
        )
        return start, peak

    def _maybe_latch_loss(self, equity: float, start: float, peak: float) -> bool:
        daily = start * (1.0 - float(self.config.limits.max_daily_loss_fraction))
        trail = peak * (1.0 - float(self.config.limits.max_trailing_drawdown_fraction))
        if equity <= daily or equity <= trail:
            if not self.kill_latched():
                self._write_kill("desk", "session_loss")
            return True
        return self.kill_latched()

    # -- orders --

    def _submit_unlocked(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float | None,
        client_order_id: str | None,
        reduce_only: bool,
    ) -> dict[str, Any]:
        self._require_armed()
        assert_live_confirmed(self.authorization.environment)
        name = symbol.strip().upper()
        order_side = _side(side)
        self._check_shape(name, qty, limit_price, reduce_only)
        client_id = self._client_id(client_order_id)
        stored = self._stored_order(client_id)
        if stored is not None:
            return {**stored, "idempotent": True, "accepted": True, **self._stamp()}
        if self.kill_latched() and not reduce_only:
            raise KillSwitchActive(str(_read_json(self.state_dir / "kill.json").get("reason")))
        now = self._now()
        if limit_price is not None and qty * limit_price > self.authorization.max_order_notional:
            return self._deny(name, order_side, qty, ("max_order_notional",), client_id)
        if self._orders_today(now) >= self.authorization.max_orders_per_day:
            return self._deny(name, order_side, qty, ("order_rate",), client_id)
        account = self._checked_account()
        positions = self.broker.get_positions()
        if reduce_only:
            self._check_reduce(name, order_side, qty, positions)
        reference = self.broker.latest_trade(name)
        basis = reference if limit_price is None else max(limit_price, reference)
        if qty * basis > self.authorization.max_order_notional:
            return self._deny(name, order_side, qty, ("max_order_notional",), client_id)
        ts_ns = _ts_ns(now)
        if self._duplicate(name, order_side, qty, basis, ts_ns):
            return self._deny(name, order_side, qty, ("duplicate_order",), client_id)
        decision = self._decide(
            name=name,
            side=order_side,
            qty=qty,
            px=reference if limit_price is None else limit_price,
            is_limit=limit_price is not None,
            positions=positions,
            reference=reference,
            equity=account.equity,
            cash=account.cash,
            now=now,
            ts_ns=ts_ns,
        )
        reasons = decision.reasons
        if not decision.allowed and not (reduce_only and set(reasons) <= _REDUCE_OVERRIDE):
            return self._deny(name, order_side, qty, reasons, client_id)
        return self._send(name, order_side, qty, limit_price, client_id, basis, ts_ns)

    def _check_shape(
        self, symbol: str, qty: float, limit_price: float | None, reduce_only: bool
    ) -> None:
        if not reduce_only and symbol not in self.authorization.symbol_allowlist:
            raise LiveRefused(f"{symbol} is not on the authorization allowlist")
        if isinstance(qty, bool) or not isinstance(qty, (int, float)) or qty <= 0.0:
            raise LiveRefused("quantity must be finite and strictly positive")
        if qty != qty or qty > float(self.config.limits.max_order_quantity):
            raise LiveRefused("quantity exceeds the pretrade maximum")
        if limit_price is not None and (
            isinstance(limit_price, bool)
            or not isinstance(limit_price, (int, float))
            or limit_price != limit_price
            or limit_price <= 0.0
        ):
            raise LiveRefused("limit price must be finite and strictly positive")

    def _check_reduce(
        self, symbol: str, side: str, qty: float, positions: tuple[PositionState, ...]
    ) -> None:
        held = 0.0
        for row in positions:
            if row.symbol == symbol:
                held = row.qty
        if side == "sell" and held <= 0.0:
            raise LiveRefused("reduce-only sell requires a long position")
        if side == "buy" and held >= 0.0:
            raise LiveRefused("reduce-only buy requires a short position")
        if qty - abs(held) > 1e-9:
            raise LiveRefused("reduce-only quantity exceeds the open position")

    def _decide(
        self,
        *,
        name: str,
        side: str,
        qty: float,
        px: float,
        is_limit: bool,
        positions: tuple[PositionState, ...],
        reference: float,
        equity: float,
        cash: float,
        now: datetime,
        ts_ns: int,
    ) -> Decision:
        start, peak = self._session_marks(equity, now)
        engine = PretradeEngine(
            self.config,
            hmac_key=self._hmac_key,
            initial_nav=equity,
            initial_cash=cash,
            ts_ns=ts_ns,
        )
        engine.update_account(
            nav=equity,
            mark_ts_ns=ts_ns,
            session_start_nav=start,
            peak_nav=peak,
        )
        symbol_id = self._install_book(engine, positions, name, reference, ts_ns)
        view = OrderView(
            symbol_id=symbol_id,
            side=1 if side == "buy" else -1,
            qty=float(qty),
            px=float(px),
            ts_ns=ts_ns,
            session_id=_business_session_id(now.astimezone(self._tz).date()),
            is_limit=1 if is_limit else 0,
            kind=KIND_ORDER,
        )
        decision = engine.decide(view, apply=False)
        if engine.book.killed:
            self._write_kill(
                "engine", "+".join(reason_names(int(engine.book.kill_reason))) or "kill"
            )
        self._maybe_latch_loss(equity, start, peak)
        return decision

    def _install_book(
        self,
        engine: PretradeEngine,
        positions: tuple[PositionState, ...],
        target: str,
        target_px: float,
        ts_ns: int,
    ) -> int:
        book: dict[str, tuple[float, float]] = {
            row.symbol: (row.qty, row.price) for row in positions
        }
        held, _price = book.get(target, (0.0, target_px))
        book[target] = (held, target_px)
        target_id = -1
        for symbol, (qty, px) in book.items():
            symbol_id = engine.ensure_symbol(symbol)
            if symbol_id < 0:
                raise LiveRefused("pretrade symbol capacity exceeded")
            engine.set_symbol(symbol_id, pos=qty, ref_px=px, ref_ts_ns=ts_ns, bid=px)
            if symbol == target:
                target_id = symbol_id
        if target_id < 0:
            raise LiveRefused("symbol was not registered")
        return target_id

    def _send(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float | None,
        client_id: str,
        basis: float,
        ts_ns: int,
    ) -> dict[str, Any]:
        try:
            ack = self.broker.submit_order(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=limit_price,
                client_order_id=client_id,
            )
        except BrokerRejected as exc:
            return self._deny(symbol, side, qty, ("broker_rejected",), client_id, error=str(exc))
        except BrokerUncertain:
            self._write_kill("desk", "submit_uncertain")
            self._audit(
                {
                    "action": "submit_uncertain",
                    "symbol": symbol,
                    "client_order_id": client_id,
                    "kill_latched": True,
                    **self._stamp(),
                }
            )
            raise
        self._remember(symbol, side, qty, basis, ts_ns, client_id, ack.broker_order_id, ack.status)
        record = {
            "action": "submit",
            "accepted": True,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "limit_price": limit_price,
            "client_order_id": client_id,
            "broker_order_id": ack.broker_order_id,
            "broker_status": ack.status,
            **self._stamp(),
        }
        self._audit(record)
        return record

    def _deny(
        self,
        symbol: str,
        side: str,
        qty: float,
        reasons: tuple[str, ...],
        client_id: str,
        *,
        error: str = "",
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "action": "deny",
            "accepted": False,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "reasons": list(reasons),
            "client_order_id": client_id,
            "broker_rejected": "broker_rejected" in reasons,
            **self._stamp(),
        }
        if error:
            record["error"] = error
        self._audit(record)
        return record

    def _flatten_unlocked(self, *, confirm: str) -> dict[str, Any]:
        if confirm != "FLATTEN":
            raise LiveRefused("flatten requires confirm=FLATTEN")
        self._require_armed()
        positions = self.broker.get_positions()
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in positions:
            if row.qty == 0.0 or row.symbol in seen:
                continue
            seen.add(row.symbol)
            side = "sell" if row.qty > 0.0 else "buy"
            results.append(
                self._submit_unlocked(
                    symbol=row.symbol,
                    side=side,
                    qty=abs(row.qty),
                    limit_price=None,
                    client_order_id=None,
                    reduce_only=True,
                )
            )
        self._write_kill("desk", "flatten")
        record = {
            "action": "flatten",
            "kill_latched": True,
            "results": results,
            **self._stamp(),
        }
        self._audit(record)
        return record

    def _cancel_unlocked(self, broker_order_id: str) -> dict[str, Any]:
        self._require_armed()
        assert_live_confirmed(self.authorization.environment)
        order_id = broker_order_id.strip()
        if not order_id:
            raise LiveRefused("broker order id is required")
        try:
            self.broker.cancel_order(order_id)
        except BrokerUncertain:
            self._write_kill("desk", "cancel_uncertain")
            self._audit(
                {
                    "action": "cancel_uncertain",
                    "broker_order_id": order_id,
                    "kill_latched": True,
                    **self._stamp(),
                }
            )
            raise
        except BrokerRejected as exc:
            record = {
                "action": "cancel_rejected",
                "accepted": False,
                "broker_rejected": True,
                "broker_order_id": order_id,
                "error": str(exc),
                **self._stamp(),
            }
            self._audit(record)
            return record
        record = {
            "action": "cancel",
            "accepted": True,
            "broker_order_id": order_id,
            **self._stamp(),
        }
        self._audit(record)
        return record

    def _cancel_all_unlocked(self) -> dict[str, Any]:
        self._require_armed()
        assert_live_confirmed(self.authorization.environment)
        try:
            self.broker.cancel_all()
        except BrokerUncertain:
            self._write_kill("desk", "cancel_uncertain")
            self._audit({"action": "cancel_all_uncertain", "kill_latched": True, **self._stamp()})
            raise
        record = {"action": "cancel_all", "accepted": True, **self._stamp()}
        self._audit(record)
        return record

    def _reconcile_unlocked(self) -> dict[str, Any]:
        self._require_armed()
        assert_live_confirmed(self.authorization.environment)
        missing: list[str] = []
        mismatched: list[str] = []
        rows = self._read_orders()
        for row in rows:
            client_id = str(row["client_order_id"])
            try:
                ack = self.broker.get_order_by_client_id(client_id)
            except BrokerRejected as exc:
                if exc.status == 404:
                    missing.append(client_id)
                    continue
                raise
            if ack.symbol and ack.symbol != str(row.get("symbol", "")).upper():
                mismatched.append(client_id)
            if ack.side and ack.side != str(row.get("side", "")):
                mismatched.append(client_id)
        record = {
            "action": "reconcile",
            "ok": not missing and not mismatched,
            "missing_at_broker": missing,
            "mismatched": mismatched,
            "n_local_orders": len(rows),
            **self._stamp(),
        }
        self._audit(record)
        return record

    # -- local order memory --

    def _orders_today(self, now: datetime) -> int:
        raw = _read_json(self.state_dir / "day_count.json")
        if raw.get("day") != now.astimezone(self._tz).date().isoformat():
            return 0
        return int(raw.get("count", 0))

    def _bump_day(self, now: datetime) -> None:
        day = now.astimezone(self._tz).date().isoformat()
        count = self._orders_today(now) + 1
        self._write_checked(self.state_dir / "day_count.json", {"day": day, "count": count})

    def _duplicate(self, symbol: str, side: str, qty: float, px: float, ts_ns: int) -> bool:
        window = int(self.config.limits.duplicate_window_ns)
        tick_q = float(self.config.limits.qty_tick)
        tick_p = float(self.config.limits.price_tick)
        raw = _read_json(self.state_dir / "recent.json")
        rows = raw.get("orders", [])
        kept: list[dict[str, Any]] = []
        hit = False
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                age = ts_ns - int(row.get("ts_ns", 0))
                if age < 0 or age > window:
                    continue
                kept.append(row)
                same = (
                    row.get("symbol") == symbol
                    and row.get("side") == side
                    and abs(float(row.get("qty", 0.0)) - qty) <= tick_q
                    and abs(float(row.get("px", 0.0)) - px) <= tick_p
                )
                hit = hit or bool(same)
        self._write_checked(self.state_dir / "recent.json", {"orders": kept})
        return hit

    def _remember(
        self,
        symbol: str,
        side: str,
        qty: float,
        px: float,
        ts_ns: int,
        client_id: str,
        broker_order_id: str,
        status: str,
    ) -> None:
        now = self._now()
        self._bump_day(now)
        recent = _read_json(self.state_dir / "recent.json")
        orders = recent.get("orders", [])
        if not isinstance(orders, list):
            orders = []
        orders.append({"symbol": symbol, "side": side, "qty": qty, "px": px, "ts_ns": ts_ns})
        self._write_checked(self.state_dir / "recent.json", {"orders": orders[-200:]})
        line = {
            "client_order_id": client_id,
            "broker_order_id": broker_order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "broker_status": status,
            "ts_ns": ts_ns,
        }
        self._append(self.state_dir / "orders.jsonl", line)

    def _stored_order(self, client_id: str) -> dict[str, Any] | None:
        for row in self._read_orders():
            if row.get("client_order_id") == client_id:
                return {
                    "action": "submit",
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "qty": row.get("qty"),
                    "client_order_id": client_id,
                    "broker_order_id": row.get("broker_order_id"),
                    "broker_status": row.get("broker_status"),
                }
        return None

    def _read_orders(self) -> list[dict[str, Any]]:
        path = self.state_dir / "orders.jsonl"
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise LiveRefused("orders journal has a non-object line")
            rows.append(raw)
        return rows

    def _client_id(self, client_order_id: str | None) -> str:
        if client_order_id is None:
            return _new_client_order_id()
        if _CLIENT_ID.fullmatch(client_order_id) is None:
            raise LiveRefused("client_order_id must be 8-48 letters, digits, _ or -")
        return client_order_id

    # -- audit --

    def _write_checked(self, path: Path, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, sort_keys=True, default=str)
        self._reject_secrets(text)
        _write_json(path, payload)

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True, default=str) + "\n"
        self._reject_secrets(line)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _audit(self, record: dict[str, Any]) -> None:
        self._append(self.state_dir / "audit.jsonl", record)

    def _reject_secrets(self, text: str) -> None:
        if self.broker.redact_text(text) != text:
            raise LiveRefused("refusing to write a record that contains broker credentials")


def open_desk(
    *,
    authorization: Path,
    pretrade: Path,
    state_dir: Path,
    transport: HttpTransport | None = None,
    clock: Callable[[], datetime] | None = None,
    environ: Mapping[str, str] | None = None,
) -> LiveDesk:
    """Build a desk from an authorization file, pre-trade config, and the environment."""
    auth = load_authorization(authorization)
    current = (clock or _wall_clock)()
    assert_current(auth, now=current)
    assert_live_confirmed(auth.environment, environ)
    key = hmac_key_from_env(environ)
    config, _digest, _signature = load_pretrade_config(pretrade, hmac_key=key)
    creds = credentials_from_env(environ)
    client = AlpacaClient(
        environment=auth.environment,
        key_id=creds.key_id,
        secret=creds.secret,
        transport=transport or UrllibTransport(environment=auth.environment),
    )
    return LiveDesk(
        authorization=auth,
        authorization_sha256=authorization_sha256(authorization),
        config=config,
        hmac_key=key,
        broker=client,
        state_dir=state_dir,
        clock=clock,
    )


def preflight_report(
    *,
    authorization_path: Path,
    pretrade_path: Path,
    state_dir: Path,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Local checks only. Does not call the broker."""
    errors: list[str] = []
    environment = "unconfigured"
    account_id = ""
    flags = credential_flags(environ)
    auth: LiveAuthorization | None = None
    try:
        auth = load_authorization(authorization_path)
        environment = auth.environment
        account_id = auth.account_id
        assert_current(auth, now=now or _wall_clock())
        assert_live_confirmed(auth.environment, environ)
    except (OSError, LiveRefused, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    try:
        key = hmac_key_from_env(environ)
        config, _digest, _signature = load_pretrade_config(pretrade_path, hmac_key=key)
        if auth is not None and auth.max_order_notional > config.limits.max_order_notional:
            errors.append("authorization max_order_notional exceeds the pretrade limit")
    except (OSError, LiveRefused, ValueError) as exc:
        errors.append(str(exc))
    try:
        credentials_from_env(environ)
    except LiveRefused as exc:
        errors.append(str(exc))
    armed = False
    try:
        state = _prepare_state_dir(state_dir)
        raw = _read_json(state / "armed.json")
        if raw.get("armed") is True and auth is not None:
            armed = raw.get("authorization_sha256") == authorization_sha256(authorization_path)
            if not armed:
                errors.append("armed file does not match the authorization file")
    except (OSError, LiveRefused) as exc:
        errors.append(str(exc))
    return {
        "ok": not errors,
        "errors": errors,
        "armed": armed,
        "account_id": account_id,
        **flags,
        **honesty_stamp(environment=environment),
    }
