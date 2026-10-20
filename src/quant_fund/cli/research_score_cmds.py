"""Coherence and concordance score commands (split from research_cmds).

Registers on the shared Typer ``app`` at import time. ``research_cmds``
imports this module so the commands stay mounted and the names stay
re-exported.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .app import app
from .support import _cfg, format_data_label


@app.command()
def coherence(
    panels: str | None = typer.Option(
        None, help="Comma-separated panel names (default: every registered generator)."
    ),
    methods: str | None = typer.Option(
        None,
        help="Comma-separated reconciliation methods (default: direct, naive_sum, "
        "independent_mc, copula_mc).",
    ),
    n_train: int = typer.Option(384, help="Leading rows used to fit margins and correlation."),
    n_eval: int = typer.Option(128, help="Trailing scored origins."),
    n_mc: int = typer.Option(512, help="Monte-Carlo paths for the copula aggregators (>=16)."),
    seed: int = typer.Option(0, help="Synthetic panel seed."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Score quantile-reconciliation methods on SYNTHETIC panels and seal a receipt.

    Compares direct / naive-sum / independent-MC / copula-MC aggregation on
    seeded synthetic factor, independent, heavy-tail and regime-copula panels.
    Reported as CRPS, PIT-KS and central-interval coverage — proper scores only,
    never market evidence and never a P&L or Sharpe claim.
    """
    from quant_fund.research.coherence import (
        METHODS,
        PANEL_GENERATORS,
        format_coherence_table,
        run_coherence,
        write_coherence_receipt,
    )

    panel_names = None if panels is None else [p.strip() for p in panels.split(",") if p.strip()]
    if panel_names is not None:
        unknown = sorted(set(panel_names) - set(PANEL_GENERATORS))
        if unknown:
            raise typer.BadParameter(
                f"unknown panel(s) {', '.join(unknown)}; "
                f"registered: {', '.join(sorted(PANEL_GENERATORS))}"
            )
    method_names = None if methods is None else [m.strip() for m in methods.split(",") if m.strip()]
    if method_names is not None:
        unknown = sorted(set(method_names) - set(METHODS))
        if unknown:
            raise typer.BadParameter(
                f"unknown method(s) {', '.join(unknown)}; registered: {', '.join(METHODS)}"
            )
    try:
        frame, receipt = run_coherence(
            panel_names,
            n_train,
            n_eval,
            seed=seed,
            n_mc=n_mc,
            methods=method_names if method_names is not None else METHODS,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_coherence_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_coherence_table(frame))
    typer.echo(f"verdict={receipt.get('verdict')}")
    typer.echo(f"receipt={path}")


@app.command()
def concordance(
    config: Path = typer.Option(Path("configs/research.yaml")),
    heads: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    alpha: float = typer.Option(0.10, help="DM elimination significance level."),
    n_boot: int = typer.Option(500, help="Stationary-bootstrap replicates."),
    block: float | None = typer.Option(
        None, help="Mean bootstrap block length (default: lane choice)."
    ),
    seed: int | None = typer.Option(None, help="Base seed (default: train.random_seed)."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Measure selection concordance across the SYNTHETIC fleet and seal a receipt.

    Does the ranking a lane picks agree with the ranking an independent
    elimination procedure produces? Reports Kendall tau, Jaccard overlap and the
    DM-eliminated set — selection diagnostics only.
    """
    from quant_fund.research.concordance import (
        format_concordance_table,
        run_concordance_eval,
        write_concordance_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    head_names = None if heads is None else [h.strip() for h in heads.split(",") if h.strip()]
    shard_names = None if shards is None else [s.strip() for s in shards.split(",") if s.strip()]
    try:
        # The lane resolves the fleet and shards itself and raises ValueError on
        # an unknown name, so validating here would build every head twice.
        frame, receipt = run_concordance_eval(
            seed=base_seed,
            n_train=n_train,
            n_eval=n_eval,
            alpha=alpha,
            n_boot=n_boot,
            block=block,
            head_names=head_names,
            shard_names=shard_names,
            taus=cfg.quantiles.levels,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_concordance_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_concordance_table(frame))
    typer.echo(f"verdict={receipt.get('verdict')}")
    typer.echo(f"receipt={path}")
