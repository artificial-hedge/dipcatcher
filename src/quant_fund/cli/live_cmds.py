"""One ``dipcatcher live`` command for the desk and the broker gateway.

No gateway config means the operator desk: shadow, a SYNTHETIC loopback, or
a drop folder. A gateway file (``starting_cash`` plus ``pretrade_config``)
selects the armed broker session. ``--root`` or ``--venue`` always means the
desk. The Alpaca live host is refused by the gateway.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
import yaml

from quant_fund.cli._app import app
from quant_fund.live.arming import ENV_ROOT, ArmingError
from quant_fund.live.desk import LiveDesk, LiveDeskError

live_app = typer.Typer(
    help=(
        "Operator desk by default (shadow, SYNTHETIC loopback, drop folder). "
        "A gateway config selects the broker session. Alpaca live is refused. "
        "No live-performance claim."
    )
)
app.add_typer(live_app, name="live")

_DESK_CONFIG = Path("configs/pretrade_risk.yaml")
_GATEWAY_CONFIG = Path("configs/live.yaml")


def _default_root() -> Path:
    override = os.environ.get(ENV_ROOT)
    if override:
        return Path(override)
    return Path.home() / ".dipcatcher" / "live"


def _resolve(root: Path | None) -> Path:
    return _default_root() if root is None else root


def _key(root: Path) -> bytes:
    path = root / "hmac.key"
    if path.exists():
        key = path.read_bytes()
        if len(key) < 16:
            raise LiveDeskError("hmac.key must be at least 16 bytes")
        return key
    root.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    return key


def _open_desk(
    root: Path,
    config_path: Path,
    *,
    cash: float | None,
    account: float | None,
    ts_ns: int | None,
) -> LiveDesk:
    from quant_fund.pretrade.config import load_pretrade_config

    key = _key(root)
    config, _digest, _sig = load_pretrade_config(config_path, hmac_key=key)
    return LiveDesk.open(
        root,
        hmac_key=key,
        config=config,
        cash=cash,
        account_value=account,
        ts_ns=ts_ns,
    )


def _emit_desk(payload: dict[str, object]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _fail_desk(exc: Exception) -> None:
    _emit_desk(
        {
            "schema": "live_desk_report.v1",
            "ok": False,
            "error": str(exc),
            "live_pnl_claim": False,
            "would_promote_live": False,
            "evidence_conditions": "blocked",
        }
    )
    raise typer.Exit(1) from exc


def _desk(
    root: Path | None,
    config: Path,
    fn: Callable[[LiveDesk], dict[str, object]],
    *,
    cash: float | None = None,
    account: float | None = None,
    ts_ns: int | None = None,
) -> None:
    try:
        desk = _open_desk(_resolve(root), config, cash=cash, account=account, ts_ns=ts_ns)
    except (LiveDeskError, ArmingError, OSError, ValueError) as exc:
        _fail_desk(exc)
        return
    try:
        try:
            _emit_desk(fn(desk))
        except (LiveDeskError, ArmingError, OSError, ValueError) as exc:
            _fail_desk(exc)
    finally:
        desk.close()


def _stamp(ts_ns: int | None) -> int:
    return time.time_ns() if ts_ns is None else ts_ns


def _gateway_file(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(raw, dict) and "starting_cash" in raw and "pretrade_config" in raw


def _use_gateway(
    config: Path | None,
    *,
    actor: str = "",
    state_dir: Path | None = None,
    venue: str = "",
    root: Path | None = None,
    limit: float | None = None,
    at: str | None = None,
    client_order_id: str | None = None,
) -> bool:
    """Desk flags win. Gateway flags or a gateway file select the broker session."""
    if venue or root is not None:
        return False
    if state_dir is not None or actor.strip() or at or client_order_id or limit is not None:
        return True
    return _gateway_file(config)


def _echo_gateway(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, sort_keys=True))


def _fail_gateway(message: str) -> None:
    _echo_gateway(
        {
            "error": message,
            "live_pnl_claim": False,
            "ok": False,
            "research_evidence": False,
        }
    )


def _invoke_gateway(action: Callable[[], dict[str, Any]]) -> None:
    from quant_fund.live.errors import LiveGateError

    try:
        payload = action()
    except (LiveGateError, OSError, ValueError, json.JSONDecodeError) as exc:
        _fail_gateway(str(exc))
        raise typer.Exit(code=1) from exc
    _echo_gateway(payload)
    if payload.get("ok") is False:
        raise typer.Exit(code=1)


def _open_gateway(config: Path | None, state_dir: Path | None) -> Any:
    from quant_fund.live.config import load_live_config
    from quant_fund.live.session import open_session

    path = _GATEWAY_CONFIG if config is None else config
    loaded = load_live_config(path)
    root = state_dir if state_dir is not None else Path(loaded.state_dir)
    return open_session(loaded, path, state_dir=root)


def _at(text: str | None) -> int | None:
    if text is None:
        return None
    from quant_fund.live.session import parse_timestamp

    return parse_timestamp(text)


def _desk_config(config: Path | None) -> Path:
    return _DESK_CONFIG if config is None else config


@live_app.callback(invoke_without_command=True)
def live_callback(ctx: typer.Context) -> None:
    """Show the default desk when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    _desk(None, _DESK_CONFIG, lambda desk: desk.status())


@live_app.command("status")
def live_status(
    root: Path | None = typer.Option(None, "--root", help="Desk directory."),
    config: Path | None = typer.Option(None, "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Print the desk, or the gateway when ``--state-dir`` or a gateway file is set."""
    if _use_gateway(config, state_dir=state_dir, root=root):

        def action() -> dict[str, Any]:
            session = _open_gateway(config, state_dir)
            payload: dict[str, Any] = session.status()
            return payload

        _invoke_gateway(action)
        return
    _desk(root, _desk_config(config), lambda desk: desk.status())


@live_app.command("arm")
def live_arm(
    venue: str = typer.Option("", "--venue", help="Desk venue: shadow, loopback, or drop."),
    confirm: str = typer.Option("", "--confirm", help="Drop requires SEND-LIVE-ORDERS."),
    root: Path | None = typer.Option(None, "--root"),
    config: Path | None = typer.Option(None, "--config"),
    ts_ns: int | None = typer.Option(None, "--ts-ns"),
    cash: float | None = typer.Option(None, "--cash"),
    account: float | None = typer.Option(None, "--account"),
    actor: str = typer.Option("", "--actor"),
    confirm_live_capital: str = typer.Option("", "--confirm-live-capital"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Arm the desk, or the gateway when ``--actor`` or a gateway file is set."""
    if _use_gateway(config, actor=actor, state_dir=state_dir, venue=venue, root=root):

        def action() -> dict[str, Any]:
            session = _open_gateway(config, state_dir)
            result: dict[str, Any] = session.arm(
                actor=actor, confirm_live_capital=confirm_live_capital
            )
            return result

        _invoke_gateway(action)
        return
    if not venue:
        raise typer.BadParameter("venue is required")
    stamp = _stamp(ts_ns)
    _desk(
        root,
        _desk_config(config),
        lambda desk: desk.arm(venue, confirm, stamp),
        cash=cash,
        account=account,
        ts_ns=stamp,
    )


@live_app.command("disarm")
def live_disarm(
    root: Path | None = typer.Option(None, "--root"),
    config: Path | None = typer.Option(None, "--config"),
    ts_ns: int | None = typer.Option(None, "--ts-ns"),
    actor: str = typer.Option("", "--actor"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Return the desk to shadow, or remove the gateway arm file."""
    if _use_gateway(config, actor=actor, state_dir=state_dir, root=root):

        def action() -> dict[str, Any]:
            session = _open_gateway(config, state_dir)
            result: dict[str, Any] = session.disarm(actor=actor)
            return result

        _invoke_gateway(action)
        return
    stamp = _stamp(ts_ns)
    _desk(root, _desk_config(config), lambda desk: desk.disarm(stamp), ts_ns=stamp)


@live_app.command("submit")
def live_submit(
    symbol: str = typer.Option(..., help="Ticker."),
    side: str = typer.Option(..., help="BUY/SELL for the desk, buy/sell for the gateway."),
    qty: float = typer.Option(..., help="Limit quantity."),
    mark: float = typer.Option(..., "--mark", help="Desk limit price, or the gateway reference."),
    root: Path | None = typer.Option(None, "--root"),
    config: Path | None = typer.Option(None, "--config"),
    ts_ns: int | None = typer.Option(None, "--ts-ns"),
    client_id: str | None = typer.Option(None, "--client-id"),
    cash: float | None = typer.Option(None, "--cash"),
    account: float | None = typer.Option(None, "--account"),
    limit_price: float | None = typer.Option(None, "--limit"),
    actor: str = typer.Option("", "--actor"),
    client_order_id: str | None = typer.Option(None, "--client-order-id"),
    at: str | None = typer.Option(None, "--at"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Submit one limit on the desk, or on the gateway when gateway flags are set."""
    if _use_gateway(
        config,
        actor=actor,
        state_dir=state_dir,
        root=root,
        limit=limit_price,
        at=at,
        client_order_id=client_order_id,
    ):
        if limit_price is None:
            raise typer.BadParameter("--limit is required for the gateway")
        gateway_limit = limit_price

        def action() -> dict[str, Any]:
            session = _open_gateway(config, state_dir)
            result: dict[str, Any] = session.submit(
                symbol=symbol,
                side=side,
                quantity=qty,
                limit_price=gateway_limit,
                mark=mark,
                actor=actor,
                client_order_id=client_order_id or client_id,
                at_ns=_at(at),
            )
            return result

        _invoke_gateway(action)
        return
    stamp = _stamp(ts_ns)

    def run(desk: LiveDesk) -> dict[str, object]:
        return desk.submit(symbol, side, qty, mark, stamp, client_order_id=client_id)

    _desk(root, _desk_config(config), run, cash=cash, account=account, ts_ns=stamp)


@live_app.command("cancel")
def live_cancel(
    order_id: str = typer.Option("", "--order-id"),
    client_order_id: str = typer.Option("", "--client-order-id"),
    root: Path | None = typer.Option(None, "--root"),
    config: Path | None = typer.Option(None, "--config"),
    ts_ns: int | None = typer.Option(None, "--ts-ns"),
    actor: str = typer.Option("", "--actor"),
    at: str | None = typer.Option(None, "--at"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Cancel a desk order, or a gateway order when ``--client-order-id`` is set."""
    if _use_gateway(
        config,
        actor=actor,
        state_dir=state_dir,
        root=root,
        at=at,
        client_order_id=client_order_id or None,
    ):

        def action() -> dict[str, Any]:
            session = _open_gateway(config, state_dir)
            result: dict[str, Any] = session.cancel(
                client_order_id=client_order_id,
                actor=actor,
                at_ns=_at(at),
            )
            return result

        _invoke_gateway(action)
        return
    if not order_id:
        raise typer.BadParameter("--order-id is required")
    stamp = _stamp(ts_ns)
    _desk(root, _desk_config(config), lambda desk: desk.cancel(order_id, stamp), ts_ns=stamp)


@live_app.command("poll")
def live_poll(
    root: Path | None = typer.Option(None, "--root"),
    config: Path | None = typer.Option(None, "--config"),
    ts_ns: int | None = typer.Option(None, "--ts-ns"),
) -> None:
    """Read inbound desk events and apply acks, fills, and cancels."""
    stamp = _stamp(ts_ns)
    _desk(root, _desk_config(config), lambda desk: desk.poll(stamp), ts_ns=stamp)


@live_app.command("kill")
def live_kill(
    reason: str = typer.Option(..., "--reason"),
    actor: str = typer.Option("", "--actor"),
    root: Path | None = typer.Option(None, "--root"),
    config: Path | None = typer.Option(None, "--config"),
    ts_ns: int | None = typer.Option(None, "--ts-ns"),
    at: str | None = typer.Option(None, "--at"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Latch the kill switch on the desk, or on the gateway."""
    if _use_gateway(config, actor=actor, state_dir=state_dir, root=root, at=at):

        def action() -> dict[str, Any]:
            session = _open_gateway(config, state_dir)
            result: dict[str, Any] = session.kill(actor=actor, reason=reason, at_ns=_at(at))
            return result

        _invoke_gateway(action)
        return
    stamp = _stamp(ts_ns)
    who = actor or "operator"
    _desk(root, _desk_config(config), lambda desk: desk.kill(who, reason, stamp), ts_ns=stamp)


@live_app.command("reset")
def live_reset(
    reason: str = typer.Option(..., "--reason"),
    actor: str = typer.Option("", "--actor"),
    root: Path | None = typer.Option(None, "--root"),
    config: Path | None = typer.Option(None, "--config"),
    ts_ns: int | None = typer.Option(None, "--ts-ns"),
    at: str | None = typer.Option(None, "--at"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Clear a desk halt, or the gateway kill latch."""
    if _use_gateway(config, actor=actor, state_dir=state_dir, root=root, at=at):

        def action() -> dict[str, Any]:
            session = _open_gateway(config, state_dir)
            result: dict[str, Any] = session.reset_kill(actor=actor, reason=reason, at_ns=_at(at))
            return result

        _invoke_gateway(action)
        return
    stamp = _stamp(ts_ns)
    who = actor or "operator"
    _desk(root, _desk_config(config), lambda desk: desk.reset(who, reason, stamp), ts_ns=stamp)


@live_app.command("flatten")
def live_flatten(
    root: Path | None = typer.Option(None, "--root"),
    config: Path | None = typer.Option(None, "--config"),
    ts_ns: int | None = typer.Option(None, "--ts-ns"),
    actor: str = typer.Option("", "--actor"),
    at: str | None = typer.Option(None, "--at"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Flatten the desk, or the gateway when gateway flags are set."""
    if _use_gateway(config, actor=actor, state_dir=state_dir, root=root, at=at):

        def action() -> dict[str, Any]:
            session = _open_gateway(config, state_dir)
            result: dict[str, Any] = session.flatten(actor=actor, at_ns=_at(at))
            return result

        _invoke_gateway(action)
        return
    stamp = _stamp(ts_ns)
    _desk(root, _desk_config(config), lambda desk: desk.flatten(stamp), ts_ns=stamp)


@live_app.command("mark")
def live_mark(
    symbol: str = typer.Option(..., "--symbol"),
    price: float = typer.Option(..., "--price"),
    actor: str = typer.Option(..., "--actor"),
    config: Path | None = typer.Option(None, "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Store a gateway reference price used later by flatten."""

    def action() -> dict[str, Any]:
        session = _open_gateway(config, state_dir)
        result: dict[str, Any] = session.mark(symbol=symbol, price=price, actor=actor)
        return result

    _invoke_gateway(action)


@live_app.command("reconcile")
def live_reconcile(
    at: str | None = typer.Option(None, "--at"),
    config: Path | None = typer.Option(None, "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Pull gateway fills and compare cash and positions to the broker snapshot."""

    def action() -> dict[str, Any]:
        session = _open_gateway(config, state_dir)
        result: dict[str, Any] = session.reconcile(at_ns=_at(at))
        return result

    _invoke_gateway(action)
