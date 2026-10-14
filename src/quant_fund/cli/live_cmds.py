"""Operator broker routing. Not registered in the model harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer

from quant_fund.live.desk import honesty_stamp, open_desk, preflight_report
from quant_fund.live.errors import BrokerRejected, BrokerUncertain, LiveRefused
from quant_fund.schemas.errors import KillSwitchActive

live_app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Route orders to the operator's own Alpaca account. "
        "Research commands never call this. Not an institutional live-trading claim."
    ),
)


def _emit(payload: dict[str, Any], *, code: int = 0) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if code:
        raise typer.Exit(code=code)


def _fail(exc: Exception, *, code: int = 2) -> NoReturn:
    payload = {
        "ok": False,
        "error": str(exc),
        **honesty_stamp(environment="unconfigured"),
    }
    _emit(payload, code=code)
    raise AssertionError("exit")


def _desk(authorization: Path, pretrade: Path, state_dir: Path) -> Any:
    try:
        return open_desk(authorization=authorization, pretrade=pretrade, state_dir=state_dir)
    except (LiveRefused, ValueError, OSError, BrokerRejected) as exc:
        _fail(exc)


@live_app.command("preflight")
def preflight(
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Check the authorization file and credentials. Does not call the broker."""
    report = preflight_report(
        authorization_path=authorization,
        pretrade_path=pretrade,
        state_dir=state_dir,
    )
    _emit(report, code=0 if report["ok"] else 2)


@live_app.command("arm")
def arm(
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Arm routing for the authorization file. Does not clear a kill latch."""
    desk = _desk(authorization, pretrade, state_dir)
    _emit(desk.arm())


@live_app.command("disarm")
def disarm(
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Stop routing. The kill latch is left as it is."""
    desk = _desk(authorization, pretrade, state_dir)
    _emit(desk.disarm())


@live_app.command("status")
def status(
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Show the local arm and kill state. Does not call the broker."""
    desk = _desk(authorization, pretrade, state_dir)
    _emit(desk.status())


@live_app.command("sync")
def sync(
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Read the broker account and positions. A session-loss breach latches the kill switch."""
    desk = _desk(authorization, pretrade, state_dir)
    try:
        _emit(desk.sync())
    except (LiveRefused, BrokerRejected, KillSwitchActive) as exc:
        _fail(exc, code=4)


@live_app.command("order")
def order(
    symbol: str = typer.Option(..., "--symbol"),
    side: str = typer.Option(..., "--side"),
    qty: float = typer.Option(..., "--qty"),
    limit_price: float | None = typer.Option(None, "--limit-price"),
    market: bool = typer.Option(False, "--market", help="Market order. Requires this flag."),
    client_order_id: str | None = typer.Option(None, "--client-order-id"),
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Submit one order. Limit is the default. A deny does not call the broker for a new order."""
    if market and limit_price is not None:
        raise typer.BadParameter("pass only one of --market and --limit-price")
    if not market and limit_price is None:
        raise typer.BadParameter("a limit price is required unless --market is set")
    desk = _desk(authorization, pretrade, state_dir)
    try:
        result = desk.submit(
            symbol=symbol,
            side=side,
            qty=qty,
            limit_price=None if market else limit_price,
            client_order_id=client_order_id,
        )
    except BrokerUncertain as exc:
        _fail(exc, code=5)
    except (LiveRefused, KillSwitchActive) as exc:
        _fail(exc)
    code = 0 if result.get("accepted") else 3
    _emit(result, code=code)


@live_app.command("cancel")
def cancel(
    broker_order_id: str = typer.Option(..., "--broker-order-id"),
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Cancel one open broker order. Allowed while the kill switch is latched."""
    desk = _desk(authorization, pretrade, state_dir)
    try:
        result = desk.cancel(broker_order_id)
    except BrokerUncertain as exc:
        _fail(exc, code=5)
    except LiveRefused as exc:
        _fail(exc)
    _emit(result, code=0 if result.get("accepted") else 4)


@live_app.command("cancel-all")
def cancel_all(
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Cancel every open order at the broker."""
    desk = _desk(authorization, pretrade, state_dir)
    try:
        _emit(desk.cancel_all())
    except BrokerUncertain as exc:
        _fail(exc, code=5)
    except LiveRefused as exc:
        _fail(exc)


@live_app.command("flatten")
def flatten(
    confirm: str = typer.Option(..., "--confirm", help="Must be the exact word FLATTEN."),
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Close open positions with reduce-only orders, then latch the kill switch."""
    desk = _desk(authorization, pretrade, state_dir)
    try:
        result = desk.flatten(confirm=confirm)
    except BrokerUncertain as exc:
        _fail(exc, code=5)
    except (LiveRefused, KillSwitchActive) as exc:
        _fail(exc)
    _emit(result)


@live_app.command("kill")
def kill(
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(..., "--reason"),
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Latch the local kill switch. Does not call the broker."""
    desk = _desk(authorization, pretrade, state_dir)
    _emit(desk.kill(actor=actor, reason=reason))


@live_app.command("reset-kill")
def reset_kill(
    actor: str = typer.Option(..., "--actor"),
    reason: str = typer.Option(..., "--reason"),
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Clear the local kill switch. Does not cancel broker orders."""
    desk = _desk(authorization, pretrade, state_dir)
    _emit(desk.reset_kill(actor=actor, reason=reason))


@live_app.command("reconcile")
def reconcile(
    authorization: Path = typer.Option(Path("configs/live_authorization.json"), "--authorization"),
    pretrade: Path = typer.Option(Path("configs/pretrade_risk.yaml"), "--pretrade"),
    state_dir: Path = typer.Option(Path("data/live"), "--state-dir"),
) -> None:
    """Match the local order journal to the broker. A missing order fails the report."""
    desk = _desk(authorization, pretrade, state_dir)
    try:
        result = desk.reconcile()
    except (LiveRefused, BrokerRejected) as exc:
        _fail(exc, code=4)
    _emit(result, code=0 if result.get("ok") else 4)
