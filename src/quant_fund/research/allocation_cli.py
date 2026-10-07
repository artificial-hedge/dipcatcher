"""``dipcatcher allocation`` sub-typer — portfolio-construction research pack.

Wraps :mod:`quant_fund.research.allocation`, which shipped fully documented and
fully tested with no operator surface at all: the causal weight engines
(inverse-volatility, risk-parity, Kelly, vol-target), the strictly causal
walk-forward evaluator, and the hash-bound receipt pair.

Honesty contract: the default input is a seeded SYNTHETIC return panel and the
output is tracking error against a vol target, weight turnover, and
equal-risk-contribution residuals — proper construction diagnostics. Never
P&L, Sharpe, or NAV; ``build_receipt`` refuses a metrics block carrying a
forbidden headline key. Supplying ``--returns`` relabels the evidence, it does
not make it market proof.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

if TYPE_CHECKING:
    # Type-only: importing the engines module at runtime would pull numpy into
    # every ``dipcatcher`` invocation, which the lazy-import convention forbids.
    from quant_fund.research.allocation.engines import EngineName

allocation_app = typer.Typer(
    help="Portfolio-construction research pack: causal weight engines, walk-forward "
    "construction diagnostics. Simulation only — never a live book."
)


def _data_label(*, synthetic: bool, data_source: str) -> str:
    """Always-printed DATA_LABEL line.

    Mirrors :func:`quant_fund.cli.support.format_data_label` exactly. It is
    duplicated rather than imported because the ``library-no-cli-imports``
    architecture guard (``configs/arch_boundaries.toml``) forbids any
    non-``quant_fund.cli`` module from importing the CLI — even lazily — and
    this sub-typer lives in the research layer like ``research100_cli.py``.
    ``tests/unit/research/test_allocation_cli.py`` pins the two in agreement so
    the copy cannot drift from the canonical formatter.
    """
    return f"DATA_LABEL={'SYNTHETIC' if synthetic else data_source}"


def _returns_panel(
    returns_path: Path | None,
    *,
    n_assets: int,
    n_periods: int,
    seed: int,
) -> tuple[Any, bool, str]:
    """Load a returns matrix, or generate a seeded SYNTHETIC one.

    Returns ``(matrix, synthetic, label)``. The label travels into the receipt
    and the printed ``DATA_LABEL`` line so the evidence is never mislabelled.
    """
    import numpy as np

    if returns_path is None:
        if n_assets < 2:
            raise ValueError("--n-assets must be >= 2")
        if n_periods < 8:
            raise ValueError("--n-periods must be >= 8")
        rng = np.random.default_rng(seed)
        loadings = rng.normal(size=(n_assets, n_assets))
        cov = loadings @ loadings.T * 1e-4 + np.diag(rng.uniform(1e-4, 4e-4, n_assets))
        panel = rng.multivariate_normal(np.zeros(n_assets), cov, n_periods)
        return np.asarray(panel, dtype=float), True, "SYNTHETIC"

    import polars as pl

    path = Path(returns_path)
    if not path.is_file():
        raise ValueError(f"returns file not found: {path}")
    frame = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)
    numeric = frame.select(
        [pl.col(c).cast(pl.Float64) for c in frame.columns if frame[c].dtype.is_numeric()]
    )
    if numeric.width < 2:
        raise ValueError(
            f"{path.name} needs at least 2 numeric return columns, found {numeric.width}"
        )
    matrix = numeric.to_numpy()
    if matrix.shape[0] < 8:
        raise ValueError(f"{path.name} needs at least 8 rows, found {matrix.shape[0]}")
    return matrix, False, path.name


def _constraints(long_only: bool, leverage_cap: float, min_weight: float, max_weight: float) -> Any:
    from quant_fund.research.allocation import AllocationConstraints

    return AllocationConstraints(
        long_only=long_only,
        leverage_cap=leverage_cap,
        min_weight=min_weight,
        max_weight=max_weight,
    )


def _resolve_engines(engines: str | None, *, target_vol: float | None = None) -> list[EngineName]:
    """Validate a comma-separated engine list against the registry.

    ``vol_target`` rescales a base engine to a vol target, so it is meaningless
    without one: it is dropped from the *default* set when ``--target-vol`` is
    absent (keeping a bare ``allocation compare`` runnable), but naming it
    explicitly without a target is an error rather than a silent skip.

    Returns ``EngineName`` rather than ``str`` so callers pass the engines
    straight to ``run_walk_forward``/``compare_engines`` unsuppressed. The cast
    is sound: every returned name is either drawn from ``ENGINE_NAMES`` or has
    just been validated as a subset of it.
    """
    from quant_fund.research.allocation import ENGINE_NAMES

    if engines is None:
        if target_vol is None:
            return cast(
                "list[EngineName]",
                [name for name in ENGINE_NAMES if name != "vol_target"],
            )
        return cast("list[EngineName]", list(ENGINE_NAMES))
    names = [e.strip() for e in engines.split(",") if e.strip()]
    unknown = sorted(set(names) - set(ENGINE_NAMES))
    if unknown:
        raise ValueError(
            f"unknown engine(s) {', '.join(unknown)}; registered: {', '.join(ENGINE_NAMES)}"
        )
    if not names:
        raise ValueError("no engines selected")
    if "vol_target" in names and target_vol is None:
        raise ValueError("engine 'vol_target' requires a positive --target-vol")
    return cast("list[EngineName]", names)


def _metrics_line(name: str, evaluation: Any) -> str:
    m = evaluation.metrics
    line = (
        f"{name:<18} rebalances={int(m['n_rebalances'])} "
        f"vol_rmse={m['vol_rmse']:.6f} vol_bias={m['vol_bias']:+.6f} "
        f"turnover_mean={m['mean_turnover']:.4f} "
        f"erc_res_mean={m['erc_residual_mean']:.4f} erc_res_max={m['erc_residual_max']:.4f}"
    )
    if "target_vol_rmse" in m:
        line += f" target_vol_rmse={m['target_vol_rmse']:.6f}"
    return line


@allocation_app.command("engines")
def engines_cmd() -> None:
    """List the registered causal weight engines."""
    from quant_fund.research.allocation import ENGINE_NAMES

    for name in ENGINE_NAMES:
        typer.echo(name)


@allocation_app.command("walk-forward")
def walk_forward_cmd(
    engine: str = typer.Option("inverse_volatility", help="Weight engine to evaluate."),
    returns: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="CSV/parquet of per-period simple returns (one column per asset). "
        "Default: a seeded SYNTHETIC panel.",
    ),
    n_assets: int = typer.Option(4, help="SYNTHETIC panel asset count."),
    n_periods: int = typer.Option(400, help="SYNTHETIC panel length in periods."),
    window: int = typer.Option(60, help="Trailing fit window (strictly causal)."),
    step: int = typer.Option(5, help="Rebalance step in periods."),
    horizon: int | None = typer.Option(
        None, help="Forward measurement block length (default: step)."
    ),
    target_vol: float | None = typer.Option(
        None, help="Annualised-style vol target; enables the vol_target rescale diagnostics."
    ),
    kelly_fraction: float = typer.Option(0.5, help="Kelly scaling fraction."),
    long_only: bool = typer.Option(
        True, "--long-only/--allow-short", help="Clip negative weights."
    ),
    leverage_cap: float = typer.Option(1.0, help="Gross exposure cap."),
    min_weight: float = typer.Option(0.0, help="Inclusion floor."),
    max_weight: float = typer.Option(1.0, help="Per-asset cap."),
    seed: int = typer.Option(1, help="SYNTHETIC panel seed."),
    out: Path | None = typer.Option(None, help="Write an allocation_evaluation.v1 receipt here."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="1 = allocation_evaluation.v1 (default), 2 = sealed receipt.v2.",
    ),
) -> None:
    """Strictly causal walk-forward evaluation of one weight engine.

    Weights at each decision use only rows strictly before it. Reports
    realized-vol-vs-predicted tracking error, turnover, and equal-risk-
    contribution residuals — construction diagnostics, never P&L or Sharpe.
    """
    from quant_fund.research.allocation import build_receipt, run_walk_forward, write_receipt

    try:
        names = _resolve_engines(engine, target_vol=target_vol)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        matrix, synthetic, label = _returns_panel(
            returns, n_assets=n_assets, n_periods=n_periods, seed=seed
        )
        constraints = _constraints(long_only, leverage_cap, min_weight, max_weight)
        evaluation = run_walk_forward(
            matrix,
            names[0],
            window=window,
            step=step,
            horizon=horizon,
            constraints=constraints,
            kelly_fraction=kelly_fraction,
            target_vol=target_vol,
        )
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(_data_label(synthetic=synthetic, data_source=label))
    typer.echo(_metrics_line(evaluation.engine, evaluation))
    typer.echo(
        f"assets={evaluation.n_assets} window={evaluation.window} step={evaluation.step} "
        f"horizon={evaluation.horizon} constraint_violations={evaluation.constraint_violations}"
    )
    if out is not None:
        if receipt_version not in (1, 2):
            raise typer.BadParameter("--receipt-version must be 1 or 2")
        receipt = build_receipt(
            evaluation,
            synthetic=synthetic,
            parameters={
                "engine": evaluation.engine,
                "window": window,
                "step": step,
                "horizon": evaluation.horizon,
                "kelly_fraction": kelly_fraction,
                "target_vol": target_vol,
                "long_only": long_only,
                "leverage_cap": leverage_cap,
                "min_weight": min_weight,
                "max_weight": max_weight,
                "seed": seed if returns is None else None,
                "returns_file": None if returns is None else str(returns),
            },
        )
        try:
            path = write_receipt(receipt, out, receipt_version=receipt_version)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"receipt={path}")


@allocation_app.command("compare")
def compare_cmd(
    engines: str | None = typer.Option(
        None,
        help="Comma-separated engine names (default: every engine runnable with the "
        "options given — vol_target is included only when --target-vol is set).",
    ),
    returns: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="CSV/parquet returns matrix."
    ),
    n_assets: int = typer.Option(4, help="SYNTHETIC panel asset count."),
    n_periods: int = typer.Option(400, help="SYNTHETIC panel length in periods."),
    window: int = typer.Option(60, help="Trailing fit window (strictly causal)."),
    step: int = typer.Option(5, help="Rebalance step in periods."),
    target_vol: float | None = typer.Option(
        None, help="Vol target; required to include the vol_target engine."
    ),
    kelly_fraction: float = typer.Option(0.5, help="Kelly scaling fraction."),
    seed: int = typer.Option(1, help="SYNTHETIC panel seed."),
    out_dir: Path | None = typer.Option(
        None, help="Write one allocation_evaluation.v1 receipt per engine here."
    ),
) -> None:
    """Run every engine on the same panel and rank them on tracking error.

    One shared SYNTHETIC panel, so the comparison is over construction rules
    rather than over different data. Ranks on realized-vol RMSE and turnover —
    never on P&L, and never a claim that any engine is best out of sample.
    """
    from quant_fund.research.allocation import build_receipt, compare_engines, write_receipt

    try:
        names = _resolve_engines(engines, target_vol=target_vol)
        matrix, synthetic, label = _returns_panel(
            returns, n_assets=n_assets, n_periods=n_periods, seed=seed
        )
        results = compare_engines(
            matrix,
            names,
            window=window,
            step=step,
            target_vol=target_vol,
            kelly_fraction=kelly_fraction,
        )
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(_data_label(synthetic=synthetic, data_source=label))
    for name in names:
        typer.echo(_metrics_line(name, results[name]))
    best = min(results, key=lambda k: results[k].metrics["vol_rmse"])
    typer.echo(
        f"lowest_vol_rmse={best} (in-sample ranking on this panel only — "
        f"not an out-of-sample claim)"
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for engine_name, evaluation in results.items():
            receipt = build_receipt(
                evaluation,
                synthetic=synthetic,
                parameters={
                    "engine": engine_name,
                    "window": window,
                    "step": step,
                    "target_vol": target_vol,
                    "kelly_fraction": kelly_fraction,
                    "seed": seed if returns is None else None,
                    "returns_file": None if returns is None else str(returns),
                },
            )
            path = write_receipt(receipt, out_dir / f"allocation_{engine_name}.json")
            typer.echo(f"receipt={path}")


@allocation_app.command("verify-receipt")
def verify_receipt_cmd(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Allocation receipt JSON."),
) -> None:
    """Re-derive an allocation receipt's payload hash. Exits non-zero on any error."""
    import json

    from quant_fund.research.allocation import verify_allocation_receipt

    result = verify_allocation_receipt(path)
    typer.echo(json.dumps(result, indent=2))
    raise typer.Exit(code=0 if result["valid"] else 1)
