"""Forecast harness, bench, and doctor commands for ``fx1``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from fx1.cli._app import app


@app.command("dipbench")
def dipbench_demo(
    data_dir: Path | None = typer.Option(
        None, help="Directory of *_1d.parquet series for a real-data run."
    ),
    out: Path | None = typer.Option(None, help="Receipt JSON path (real-data run)."),
    threshold: float = typer.Option(0.10, help="Drawdown depth that defines a dip."),
) -> None:
    """Dip Quality Score bench. Without --data-dir: smoke the bench on a
    built-in synthetic series (SYNTHETIC — correctness only, not market
    evidence). With --data-dir: full receipt over real historical bars."""
    if data_dir is not None:
        from fx1.bench.run import run_dip_bench

        receipt = run_dip_bench(data_dir, threshold=threshold)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        typer.echo(
            json.dumps(
                {
                    "label": "research/backtest evidence, not live performance",
                    "events": receipt["n_events"],
                    "baseline_recovery": receipt["baseline_recovery"],
                    "receipt": str(out) if out else None,
                },
                indent=2,
            )
        )
        return
    from fx1.bench.dip import (
        DipForecast,
        detect_dip_events,
        evaluate_forecasts,
        unconditional_baseline,
    )

    closes = [100.0, 102.0, 88.0, 90.0, 103.0, 104.0, 92.0, 95.0, 106.0]
    dates = [f"2026-01-{i + 1:02d}" for i in range(len(closes))]
    events = detect_dip_events(closes, dates, "SYNTHETIC", threshold=0.10, horizons_bars={"1m": 3})
    baseline = unconditional_baseline(events)
    forecasts = [DipForecast(e.asset, e.trough_date, {"1m": 0.8}) for e in events]
    metrics = evaluate_forecasts(events, forecasts)
    typer.echo(
        json.dumps(
            {"label": "SYNTHETIC", "events": len(events), "baseline": baseline, "metrics": metrics},
            indent=2,
        )
    )


@app.command("infer")
def infer_cmd(
    config: Path = typer.Option(..., "--config", help="Harness YAML or JSON config."),
) -> None:
    """Run batch or walk-forward inference and write a forecast parquet.

    The fx-1 forecaster is external: ``model.name=fx-1`` requires
    ``model.entrypoint``. Reference names ``dummy-zero`` and ``dummy-momentum``
    are not fx-1. This command does not train and does not place orders.
    """
    from fx1.forecast.config import load_harness_config
    from fx1.forecast.runner import run_inference

    result = run_inference(load_harness_config(config))
    typer.echo(
        json.dumps(
            {
                "forecasts": str(result.parquet_path),
                "metadata": str(result.meta_path),
                "n_rows": result.n_rows,
                "model_name": result.metadata["model_name"],
                "model_role": result.metadata["model_role"],
                "data_label": result.metadata["data_label"],
                "research_only": True,
                "live_pnl_claim": False,
            },
            indent=2,
        )
    )


@app.command("backtest")
def backtest_cmd(
    config: Path = typer.Option(..., "--config", help="Harness YAML or JSON config."),
    forecasts: Path | None = typer.Option(
        None,
        "--forecasts",
        help="Forecast parquet. Defaults to inference.output_parquet in the config.",
    ),
) -> None:
    """Score forecasts and a placeholder signal map.

    Reports forecast scores (IC, rank IC, hit rate, MAE, RMSE) and research
    diagnostics of the placeholder mapping. Does not place orders. Run
    ``fx1 infer`` first when the forecast parquet is not already on disk.
    """
    from fx1.forecast.config import load_harness_config
    from fx1.forecast.runner import run_signal_evaluation

    cfg = load_harness_config(config)
    frame = None
    if forecasts is not None:
        import polars as pl

        frame = pl.read_parquet(forecasts)
    report = run_signal_evaluation(cfg, forecasts=frame)
    typer.echo(json.dumps(report, indent=2))


@app.command("doctor")
def doctor(root: Path = typer.Option(Path("."), help="Repo root to inspect.")) -> None:
    """fx-1 readiness status (presence flags only — never secret values)."""
    from fx1.doctor import collect_status

    typer.echo(json.dumps(collect_status(root), indent=2))
