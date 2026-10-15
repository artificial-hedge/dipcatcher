"""Fleet eval / comparison research commands (split from research_cmds).

Registers on the shared Typer ``app`` at import time. Import this module from
``cli.main`` so the commands stay mounted; do not re-define them on research_cmds.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .app import app
from .support import _cfg, format_data_label


@app.command("multih-fleet")
def multih_fleet_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    horizons: str = typer.Option("1,5,20", help="Comma-separated h-step horizons (>=1)."),
    n_train: int = typer.Option(200, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(40, help="Trailing scored origins per horizon."),
    n: int = typer.Option(400, help="Synthetic series length per shard."),
    seed: int | None = typer.Option(None, help="Base seed (default: train.random_seed)."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Race h-step quantile constructions on the SYNTHETIC fleet and seal a receipt.

    For each head and horizon, scores the native multi-horizon fit against the
    iid-sqrt and empirical-ratio h-step extensions. Pinball, central-interval
    coverage and Kupiec p — proper scores only, never market evidence.
    """
    from quant_fund.research.fleet_eval import resolve_shard_generators
    from quant_fund.research.multih_fleet import (
        multih_factories,
        run_multih_fleet_eval,
        write_multih_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    taus = list(cfg.quantiles.levels)
    model_names = None if models is None else [m.strip() for m in models.split(",") if m.strip()]
    shard_names = None if shards is None else [s.strip() for s in shards.split(",") if s.strip()]
    try:
        horizon_list = sorted({int(h) for h in horizons.split(",") if h.strip()})
    except ValueError as exc:
        raise typer.BadParameter(f"--horizons must be comma-separated integers: {exc}") from exc
    if not horizon_list or min(horizon_list) < 1:
        raise typer.BadParameter("--horizons must contain at least one horizon >= 1")
    try:
        factories = multih_factories(taus, base_seed, names=model_names)
        generators = resolve_shard_generators(shard_names)
        rows, receipt = run_multih_fleet_eval(
            factories,
            generators,
            taus=taus,
            horizons=horizon_list,
            n_train=n_train,
            n_eval=n_eval,
            n=n,
            seed=base_seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_multih_receipt(receipt, out_dir)
    ok = sum(1 for r in rows if r.status == "ok")
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"rows={len(rows)} ok={ok} errors={len(rows) - ok} "
        f"heads={len(factories)} shards={len(generators)} "
        f"horizons={','.join(map(str, horizon_list))}"
    )
    typer.echo(f"verdict={receipt.get('verdict')}")
    typer.echo(f"receipt={path}")


@app.command("expert-mixture")
def expert_mixture_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    heads: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    alpha: float = typer.Option(0.05, help="Fixed-share switching rate."),
    seed: int | None = typer.Option(None, help="Base seed (default: train.random_seed)."),
    dev: bool = typer.Option(False, "--dev", help="Acknowledge dev-only use; required to run."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Run prediction-with-expert-advice over the SYNTHETIC fleet and seal a receipt.

    Mixes the fleet under uniform, exponentially-weighted-average and
    fixed-share weighting. Scored on regret against the best fixed expert in
    hindsight — a mixture is never claimed dominant, and nothing here is market
    evidence.
    """
    if not dev:
        raise typer.BadParameter(
            "expert-mixture is dev-only evidence tooling; pass --dev to acknowledge."
        )
    from quant_fund.research.expert_mixture import (
        MIXERS,
        format_expert_mixture_table,
        run_expert_mixture_eval,
        write_expert_mixture_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    head_names = None if heads is None else [h.strip() for h in heads.split(",") if h.strip()]
    shard_names = None if shards is None else [s.strip() for s in shards.split(",") if s.strip()]
    try:
        # As in `concordance`: the lane resolves and validates the fleet itself.
        frame, receipt = run_expert_mixture_eval(
            seed=base_seed,
            n_train=n_train,
            n_eval=n_eval,
            alpha=alpha,
            head_names=head_names,
            shard_names=shard_names,
            taus=cfg.quantiles.levels,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_expert_mixture_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_expert_mixture_table(frame))
    typer.echo(f"mixers={','.join(MIXERS)}")
    typer.echo(f"verdict={receipt.get('verdict')}")
    typer.echo(f"receipt={path}")


@app.command()
def compare(
    run_a: Path = typer.Argument(..., exists=True, help="Receipt JSON or result dir for run A."),
    run_b: Path = typer.Argument(..., exists=True, help="Receipt JSON or result dir for run B."),
    higher_is_better: bool = typer.Option(
        False, help="Score direction: positive delta favors A (default: loss convention)."
    ),
    alpha: float = typer.Option(0.05, help="Paired bootstrap significance level."),
    n_boot: int = typer.Option(2000, help="Paired bootstrap replicates."),
    seed: int = typer.Option(7, help="Bootstrap seed."),
    min_paired: int = typer.Option(
        10, help="Below this paired n the verdict is 'insufficient paired observations'."
    ),
    series: str | None = typer.Option(
        None, help="Comma-separated series keys to restrict the comparison to."
    ),
    fmt: str = typer.Option("markdown", "--format", help="markdown or json."),
    out: Path | None = typer.Option(None, help="Write the report here instead of stdout."),
    receipt_out: Path | None = typer.Option(None, help="Also write a run_compare receipt here."),
    receipt_version: int = typer.Option(
        1, "--receipt-version", help="1 = run_compare.v1 blob (default), 2 = sealed receipt.v2."
    ),
) -> None:
    """Paired comparison of two stored research runs.

    Same lane as ``python -m quant_fund.research.compare`` (one shared
    implementation, so the two surfaces cannot drift). Diagnostic only: proper
    scores on stored receipts, never a live-trading or P&L claim. The envelope
    is labeled ``DATA_LABEL=UNKNOWN`` because a comparison of two stored runs
    cannot know whether either was SYNTHETIC.
    """
    from quant_fund.research.compare import (
        COMPARE_DATA_LABEL,
        build_compare_receipt,
        compare_runs,
    )

    if fmt not in ("markdown", "json"):
        raise typer.BadParameter("--format must be markdown or json")
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    series_keys = None if series is None else [s.strip() for s in series.split(",") if s.strip()]
    try:
        comparison = compare_runs(
            run_a,
            run_b,
            higher_is_better=higher_is_better,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            min_paired=min_paired,
            series_keys=series_keys,
        )
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    text = (
        json.dumps(comparison.to_dict(), indent=2, allow_nan=False)
        if fmt == "json"
        else comparison.to_markdown()
    )
    if out is not None:
        out.write_text(text + "\n")
        typer.echo(f"report={out}")
    else:
        typer.echo(text)
    if receipt_out is not None:
        document = build_compare_receipt(comparison, version=receipt_version)
        receipt_out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        typer.echo(f"receipt={receipt_out}")
    typer.echo(format_data_label(synthetic=False, data_source=COMPARE_DATA_LABEL))
    counts = comparison.verdict_counts
    typer.echo(
        f"series={len(comparison.series)} "
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
