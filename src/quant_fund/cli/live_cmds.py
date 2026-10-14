"""Armed order commands. Disarmed until ``live arm``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from quant_fund.cli._app import app

live_app = typer.Typer(
    help=(
        "Armed order gateway. Disarmed until `live arm`. "
        "A fill is broker accounting, not a research result and not a profit claim."
    ),
    no_args_is_help=True,
)
app.add_typer(live_app, name="live")


def _echo(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, sort_keys=True))


def _fail(message: str) -> None:
    _echo(
        {
            "error": message,
            "live_pnl_claim": False,
            "ok": False,
            "research_evidence": False,
        }
    )


def _invoke(action: Callable[[], dict[str, Any]]) -> None:
    from quant_fund.live.errors import LiveGateError

    try:
        payload = action()
    except (LiveGateError, OSError, ValueError, json.JSONDecodeError) as exc:
        _fail(str(exc))
        raise typer.Exit(code=1) from exc
    _echo(payload)
    if payload.get("ok") is False:
        raise typer.Exit(code=1)


def _open(config: Path, state_dir: Path | None) -> Any:
    from quant_fund.live.config import load_live_config
    from quant_fund.live.session import open_session

    loaded = load_live_config(config)
    root = state_dir if state_dir is not None else Path(loaded.state_dir)
    return open_session(loaded, config, state_dir=root)


def _at(text: str | None) -> int | None:
    if text is None:
        return None
    from quant_fund.live.session import parse_timestamp

    return parse_timestamp(text)


@live_app.command("status")
def live_status(
    config: Path = typer.Option(Path("configs/live.yaml"), "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Show arm, kill, and local cash/position accounting. Does not call a broker."""

    def action() -> dict[str, Any]:
        session = _open(config, state_dir)
        payload: dict[str, Any] = session.status()
        return payload

    _invoke(action)


@live_app.command("arm")
def live_arm(
    actor: str = typer.Option(..., "--actor"),
    confirm_live_capital: str = typer.Option("", "--confirm-live-capital"),
    config: Path = typer.Option(Path("configs/live.yaml"), "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Arm this state directory for the venue named in the config."""

    def action() -> dict[str, Any]:
        session = _open(config, state_dir)
        result: dict[str, Any] = session.arm(actor=actor, confirm_live_capital=confirm_live_capital)
        return result

    _invoke(action)


@live_app.command("disarm")
def live_disarm(
    actor: str = typer.Option(..., "--actor"),
    config: Path = typer.Option(Path("configs/live.yaml"), "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Remove the arm file. Open positions are left unchanged."""

    def action() -> dict[str, Any]:
        session = _open(config, state_dir)
        result: dict[str, Any] = session.disarm(actor=actor)
        return result

    _invoke(action)


@live_app.command("kill")
def live_kill(
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(..., "--reason"),
    at: str | None = typer.Option(
        None, "--at", help="Aware ISO timestamp. Refused for alpaca-live."
    ),
    config: Path = typer.Option(Path("configs/live.yaml"), "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Latch the pre-trade kill switch. This blocks new orders from this gateway."""

    def action() -> dict[str, Any]:
        session = _open(config, state_dir)
        result: dict[str, Any] = session.kill(actor=actor, reason=reason, at_ns=_at(at))
        return result

    _invoke(action)


@live_app.command("reset")
def live_reset(
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(..., "--reason"),
    at: str | None = typer.Option(None, "--at"),
    config: Path = typer.Option(Path("configs/live.yaml"), "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Clear the kill latch. Arming is unchanged."""

    def action() -> dict[str, Any]:
        session = _open(config, state_dir)
        result: dict[str, Any] = session.reset_kill(actor=actor, reason=reason, at_ns=_at(at))
        return result

    _invoke(action)


@live_app.command("mark")
def live_mark(
    symbol: str = typer.Option(..., "--symbol"),
    price: float = typer.Option(..., "--price"),
    actor: str = typer.Option(..., "--actor"),
    config: Path = typer.Option(Path("configs/live.yaml"), "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Store a reference price used later by flatten."""

    def action() -> dict[str, Any]:
        session = _open(config, state_dir)
        result: dict[str, Any] = session.mark(symbol=symbol, price=price, actor=actor)
        return result

    _invoke(action)


@live_app.command("submit")
def live_submit(
    symbol: str = typer.Option(..., "--symbol"),
    side: str = typer.Option(..., "--side"),
    qty: float = typer.Option(..., "--qty"),
    limit_price: float = typer.Option(..., "--limit"),
    mark: float = typer.Option(
        ..., "--mark", help="Reference price for the collar and exposure checks."
    ),
    actor: str = typer.Option(..., "--actor"),
    client_order_id: str | None = typer.Option(None, "--client-order-id"),
    at: str | None = typer.Option(
        None, "--at", help="Aware ISO timestamp. Refused for alpaca-live."
    ),
    config: Path = typer.Option(Path("configs/live.yaml"), "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Submit one limit order if the session is armed and pre-trade allows it."""

    def action() -> dict[str, Any]:
        session = _open(config, state_dir)
        result: dict[str, Any] = session.submit(
            symbol=symbol,
            side=side,
            quantity=qty,
            limit_price=limit_price,
            mark=mark,
            actor=actor,
            client_order_id=client_order_id,
            at_ns=_at(at),
        )
        return result

    _invoke(action)


@live_app.command("cancel")
def live_cancel(
    client_order_id: str = typer.Option(..., "--client-order-id"),
    actor: str = typer.Option(..., "--actor"),
    at: str | None = typer.Option(None, "--at"),
    config: Path = typer.Option(Path("configs/live.yaml"), "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Cancel a working order. A fill that already happened is left in place."""

    def action() -> dict[str, Any]:
        session = _open(config, state_dir)
        result: dict[str, Any] = session.cancel(
            client_order_id=client_order_id,
            actor=actor,
            at_ns=_at(at),
        )
        return result

    _invoke(action)


@live_app.command("reconcile")
def live_reconcile(
    at: str | None = typer.Option(None, "--at"),
    config: Path = typer.Option(Path("configs/live.yaml"), "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Pull fills and compare cash and positions to the broker snapshot."""

    def action() -> dict[str, Any]:
        session = _open(config, state_dir)
        result: dict[str, Any] = session.reconcile(at_ns=_at(at))
        return result

    _invoke(action)


@live_app.command("flatten")
def live_flatten(
    actor: str = typer.Option(..., "--actor"),
    at: str | None = typer.Option(None, "--at"),
    config: Path = typer.Option(Path("configs/live.yaml"), "--config"),
    state_dir: Path | None = typer.Option(None, "--state-dir"),
) -> None:
    """Post a limit at the stored mark for each open position."""

    def action() -> dict[str, Any]:
        session = _open(config, state_dir)
        result: dict[str, Any] = session.flatten(actor=actor, at_ns=_at(at))
        return result

    _invoke(action)
