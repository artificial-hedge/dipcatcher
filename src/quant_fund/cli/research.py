"""Small research-only CLI surfaces for frozen model families.

These commands intentionally expose specifications and deterministic decision
functions only. They do not connect to a broker, submit orders, or make live
P&L claims.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import typer

from quant_fund.lightspeed.specs import frozen_families
from quant_fund.quant_models.black_scholes import bs_price
from quant_fund.quant_models.gex import last_hour_decide

ls_app = typer.Typer(
    help=(
        "Lightspeed research surfaces. All commands are paper/research-only; "
        "they are not live and do not promote a strategy."
    )
)
qm_app = typer.Typer(
    help=(
        "Quant-model research calculations. These are decision/pricing helpers "
        "with no broker integration or live P&L claim."
    )
)


def _emit(payload: dict[str, Any]) -> None:
    """Emit stable JSON without serializing NumPy scalar objects."""
    typer.echo(json.dumps(payload, sort_keys=True, allow_nan=False))


@ls_app.command("specs")
def ls_specs() -> None:
    """Print the frozen Lightspeed family specifications."""
    _emit(frozen_families())


@ls_app.command("demo")
def ls_demo(
    seed: int = typer.Option(7, help="Deterministic preview seed; no market simulation is run."),
) -> None:
    """Print a deterministic, research-only family preview.

    ``seed`` is retained as an explicit receipt field for reproducible callers,
    but this command deliberately does not turn synthetic output into a P&L
    or promotion claim.
    """
    card = frozen_families()
    card.update(
        {
            "command": "ls demo",
            "seed": int(seed),
            "synthetic": True,
            "simulated_only": True,
            "note": (
                "Specification preview only; no market-data backtest, broker, "
                "live P&L, or promotion decision is produced."
            ),
        }
    )
    _emit(card)


@ls_app.command("race")
def ls_race() -> None:
    """Describe the gated challenger race (not live; risk gates remain required)."""
    _emit(
        {
            "command": "ls race",
            "research_only": True,
            "live_disabled": True,
            "live_pnl_claim": False,
            "promotion": "disabled",
            "status": "Not live: risk gates and holdout confirmation are required.",
        }
    )


@ls_app.command("confirm")
def ls_confirm() -> None:
    """Describe frozen holdout confirmation; this is not a promotion command."""
    _emit(
        {
            "command": "ls confirm",
            "research_only": True,
            "live_disabled": True,
            "live_pnl_claim": False,
            "holdout_start": "2025-01-02",
            "selection_end": "2024-12-31",
            "promotion": "disabled",
            "status": "Holdout confirmation only. Not a promotion.",
        }
    )


def _positive(name: str, value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise typer.BadParameter(f"{name} must be finite and positive")
    return float(value)


@qm_app.command("bs-price")
def qm_bs_price(
    spot: float = typer.Option(..., "--spot", help="Underlying spot price."),
    strike: float = typer.Option(..., "--strike", help="Option strike."),
    maturity: float = typer.Option(1.0, "--maturity", "--tau", help="Years to expiry."),
    rate: float = typer.Option(0.04, "--rate", help="Continuously compounded risk-free rate."),
    dividend: float = typer.Option(0.0, "--dividend", "--q", help="Continuous dividend yield."),
    volatility: float = typer.Option(0.20, "--volatility", "--sigma", help="Implied volatility."),
    option_type: str = typer.Option("call", "--option-type", "--kind"),
) -> None:
    """Calculate a BSM price for research; never a live valuation or order."""
    spot = _positive("spot", spot)
    strike = _positive("strike", strike)
    maturity = _positive("maturity", maturity)
    volatility = _positive("volatility", volatility)
    kind = option_type.lower()
    if kind not in {"call", "put"}:
        raise typer.BadParameter("option-type must be 'call' or 'put'")
    value = float(bs_price(spot, strike, maturity, rate, dividend, volatility, kind))
    _emit(
        {
            "command": "qm bs-price",
            "price": value,
            "option_type": kind,
            "research_only": True,
            "live_pnl_claim": False,
            "broker": None,
        }
    )


@qm_app.command("gex-decide")
def qm_gex_decide(
    gex: float = typer.Option(..., "--gex", help="Prior-session dealer gamma exposure."),
    sofar: float = typer.Option(..., "--sofar", help="Session return so far."),
    spot: float = typer.Option(..., "--spot", help="MES reference spot."),
    equity: float = typer.Option(1_000_000.0, "--equity"),
    leverage: float = typer.Option(1.0, "--leverage"),
) -> None:
    """Return the no-broker last-hour GEX decision."""
    spot = _positive("spot", spot)
    equity = _positive("equity", equity)
    leverage = _positive("leverage", leverage)
    decision = last_hour_decide(gex, sofar, equity, spot, leverage=leverage)
    payload = asdict(decision)
    payload.update(
        {
            "command": "qm gex-decide",
            "research_only": True,
            "live_pnl_claim": False,
            "broker": None,
        }
    )
    _emit(payload)
