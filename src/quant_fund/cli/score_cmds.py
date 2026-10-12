"""``score-forecasts`` lab command.

Importing this module registers the command on the shared Typer app. Scoring
libraries stay inside the command body so ``import quant_fund.cli.main`` does
not load numpy, polars, or scipy.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quant_fund.cli.app import app


def _format_score(value: float) -> str:
    return format(value, ".17g")


def _emit_profile(data: Path) -> None:
    from quant_fund.research.forecast_scoring import inspect_dataset

    profile = inspect_dataset(data)
    typer.echo(json.dumps(profile, indent=2, sort_keys=True, allow_nan=False))


def _emit_success(
    data_label: str, scores: dict[str, float], receipt_path: Path, *, warn_pit: bool
) -> None:
    from quant_fund.cli.support import format_data_label

    synthetic = data_label == "SYNTHETIC"
    typer.echo(format_data_label(synthetic=synthetic, data_source=data_label))
    if warn_pit:
        typer.echo(
            "WARNING: forecast issue time was not provided; point-in-time validity is NOT verified."
        )
    for name, value in scores.items():
        typer.echo(f"score.{name}={_format_score(value)}")
    typer.echo(f"receipt={receipt_path}")


def _run_score(
    *,
    task: str,
    data: Path,
    time_col: str,
    realized_col: str,
    forecast_col: str | None,
    quantile_cols: str | None,
    available_time_col: str | None,
    out_dir: Path,
    seed: int,
    receipt_version: int,
    inspect_only: bool,
) -> None:
    if inspect_only:
        _emit_profile(data)
        return
    from quant_fund.research.forecast_scoring import score_forecasts

    outcome = score_forecasts(
        data,
        task=task,
        out_dir=out_dir,
        time_col=time_col,
        realized_col=realized_col,
        forecast_col=forecast_col,
        quantile_cols=quantile_cols,
        available_time_col=available_time_col,
        seed=seed,
        receipt_version=receipt_version,
    )
    _emit_success(
        outcome.data_label,
        outcome.scores,
        outcome.receipt_path,
        warn_pit=available_time_col is None,
    )


@app.command("score-forecasts")
def score_forecasts_command(
    task: str = typer.Option(..., "--task", help="volatility, quantile, or probability."),
    data: Path = typer.Option(
        ..., "--data", help="CSV or Parquet file of forecasts and realizations."
    ),
    time_col: str = typer.Option(
        "timestamp", "--time-col", help="Realized-observation time column."
    ),
    realized_col: str = typer.Option(
        "realized", "--realized-col", help="Realized variance, value, or 0/1 outcome."
    ),
    forecast_col: str | None = typer.Option(
        None,
        "--forecast-col",
        help="Variance forecast (default forecast_var) or probability (default forecast_prob).",
    ),
    quantile_cols: str | None = typer.Option(
        None,
        "--quantile-cols",
        help="tau=column pairs, e.g. q05=low,q50=mid,q95=high. Default: columns named q plus digits.",
    ),
    available_time_col: str | None = typer.Option(
        None,
        "--available-time-col",
        help="When the forecast was issued. Must be strictly earlier than the realized time.",
    ),
    out_dir: Path = typer.Option(
        Path("receipts"), "--out-dir", help="Directory for the sealed receipt."
    ),
    seed: int = typer.Option(0, "--seed", help="Bootstrap seed in 0..2147483647."),
    receipt_version: int = typer.Option(
        2, "--receipt-version", help="Must be 2 (sealed receipt.v2)."
    ),
    inspect_only: bool = typer.Option(
        False,
        "--inspect",
        help="Print a dataset profile JSON and write nothing.",
    ),
) -> None:
    """Score user-supplied forecasts with proper scores and seal a receipt.

    Research diagnostics only. The receipt is not a live-trading claim.
    """
    from quant_fund.research.forecast_scoring import ForecastScoringError

    try:
        _run_score(
            task=task,
            data=data,
            time_col=time_col,
            realized_col=realized_col,
            forecast_col=forecast_col,
            quantile_cols=quantile_cols,
            available_time_col=available_time_col,
            out_dir=out_dir,
            seed=seed,
            receipt_version=receipt_version,
            inspect_only=inspect_only,
        )
    except ForecastScoringError as exc:
        raise typer.BadParameter(str(exc)) from None


def register_score_commands(application: typer.Typer | None = None) -> None:
    """Register ``score-forecasts`` if the import-time decorator has not already."""
    target = app if application is None else application
    names = {command.name for command in target.registered_commands}
    if "score-forecasts" in names:
        return
    target.command("score-forecasts")(score_forecasts_command)
