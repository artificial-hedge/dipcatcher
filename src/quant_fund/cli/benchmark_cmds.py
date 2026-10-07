"""Real-data benchmark, net-return tournament, and ranker-probability commands.

These three lanes were the last research modules with an operator-facing entry
point that ``dipcatcher --help`` did not show. They were reachable in the import
graph only incidentally — ``phase1_verify`` imports ``net_tournament`` to hash it
for receipt provenance — so an import-closure check counted them as wired while
no command could actually run them. Each was invocable only as
``python -m quant_fund.research.<module>``.

Wiring approach differs per lane, deliberately:

- ``real_benchmark`` and ``net_tournament`` expose clean library functions
  (``prepare_benchmark``/``score_benchmark``, ``prepare_tournament``/
  ``run_tournament``), so the commands call those directly — the same functions
  the argparse ``main()`` calls, so the two surfaces cannot drift.
- ``ranker_probability.main(argv)`` is a ~90-line orchestration body with no
  reusable entry point. Rather than duplicate it, the command translates typed
  options into an argv list and delegates, keeping exactly one implementation.

The argparse mains and their subprocess contract tests
(``tests/unit/hedge_lab/test_net_tournament.py``,
``tests/unit/core/test_real_benchmark.py``,
``tests/unit/test_forward_shadow.py``) are left untouched.

Honesty contract: these lanes score REAL data, so the DATA_LABEL reflects the
input rather than claiming SYNTHETIC, and ``ranker_probability`` stamps its own
``data_scope`` / ``holdout_previously_inspected_or_unverified`` caveats into the
receipt. Nothing here is a live-trading or performance claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

net_tournament_app = typer.Typer(
    help="Net-of-cost return tournament over a frozen real-data benchmark run."
)
real_benchmark_app = typer.Typer(
    help="Real-data benchmark: audit data and freeze a protocol, then score it."
)

#: Keys the argparse mains strip before printing — the payloads are large and
#: the operator wants the verdict, not the frozen scenario table.
_HEAVY_KEYS = frozenset({"scenarios", "benchmark_manifest"})


def _emit(result: Any, *, drop_heavy: bool = True) -> None:
    """Print a result document, minus the bulky frozen inputs by default."""
    if drop_heavy and isinstance(result, dict):
        result = {key: value for key, value in result.items() if key not in _HEAVY_KEYS}
    typer.echo(json.dumps(result, indent=2, allow_nan=False, default=str))


def _check_receipt_version(receipt_version: int) -> None:
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")


# --------------------------------------------------------------------------- #
# real_benchmark
# --------------------------------------------------------------------------- #


@real_benchmark_app.command("prepare")
def real_benchmark_prepare_cmd(
    protocol: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        help="Benchmark protocol JSON (dataset_path is relative to it).",
    ),
    output: Path = typer.Option(..., help="Run directory to create and freeze the manifest into."),
    receipt_version: int = typer.Option(
        1, "--receipt-version", help="1 = strict-digest manifest (default), 2 = sealed receipt.v2."
    ),
) -> None:
    """Audit the data and freeze a protocol without publishing any score.

    The freeze step is what makes the later score meaningful: the code, runtime
    and dataset digests are bound before anyone has seen a result. ``score``
    refuses to run if the code has changed since.
    """
    from quant_fund.cli.support import format_data_label
    from quant_fund.research.real_benchmark import prepare_benchmark

    _check_receipt_version(receipt_version)
    try:
        result = prepare_benchmark(protocol, output, receipt_version=receipt_version)
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=protocol.name))
    _emit(result)


@real_benchmark_app.command("score")
def real_benchmark_score_cmd(
    run: Path = typer.Option(
        ..., exists=True, file_okay=False, help="Frozen run directory containing manifest.json."
    ),
    phase: str = typer.Option(..., help="validation or test."),
    receipt_version: int = typer.Option(
        1, "--receipt-version", help="1 = strict-digest report (default), 2 = sealed receipt.v2."
    ),
) -> None:
    """Score a frozen run — validation first, then the declared test phase.

    Fails closed if the scoring code changed since ``prepare`` froze it.
    """
    from quant_fund.cli.support import format_data_label
    from quant_fund.research.real_benchmark import score_benchmark

    _check_receipt_version(receipt_version)
    if phase not in ("validation", "test"):
        raise typer.BadParameter("--phase must be validation or test")
    try:
        result = score_benchmark(run, phase, receipt_version=receipt_version)
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=run.name))
    _emit(result)


# --------------------------------------------------------------------------- #
# net_tournament
# --------------------------------------------------------------------------- #


@net_tournament_app.command("prepare")
def net_tournament_prepare_cmd(
    benchmark_run: Path = typer.Option(
        ..., exists=True, file_okay=False, help="Frozen real-benchmark run directory."
    ),
    spec: Path = typer.Option(..., exists=True, dir_okay=False, help="Tournament spec JSON."),
    output: Path = typer.Option(..., help="Tournament run directory to create."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="1 = strict-digest manifest (default), 2 = sealed receipt.v2 (rejected when the "
        "spec embeds headline-metric keys; run-phase reports stay v1).",
    ),
) -> None:
    """Freeze a net-of-cost tournament against a benchmark run.

    Requires the benchmark's own code and runtime digests to still match, so a
    tournament cannot be prepared against a stale or edited benchmark.
    """
    from quant_fund.cli.support import format_data_label
    from quant_fund.research.net_tournament import prepare_tournament

    _check_receipt_version(receipt_version)
    try:
        result = prepare_tournament(benchmark_run, spec, output, receipt_version=receipt_version)
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=spec.name))
    _emit(result)


@net_tournament_app.command("run")
def net_tournament_run_cmd(
    run: Path = typer.Option(
        ..., exists=True, file_okay=False, help="Frozen tournament run directory."
    ),
    phase: str = typer.Option(..., help="validation or test."),
) -> None:
    """Run one tournament phase. ``test`` requires a matching validation report.

    Fails closed if the tournament code or runtime changed since the manifest
    was frozen, or if a test phase is attempted without a validation report
    bound to this manifest's digest.
    """
    from quant_fund.cli.support import format_data_label
    from quant_fund.research.net_tournament import run_tournament

    if phase not in ("validation", "test"):
        raise typer.BadParameter("--phase must be validation or test")
    try:
        result = run_tournament(run, phase)
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=run.name))
    _emit(result)


# --------------------------------------------------------------------------- #
# ranker_probability
# --------------------------------------------------------------------------- #

ranker_probability_app = typer.Typer(
    help="Ranker probability calibration experiment over a purged train/cal/test split."
)


@ranker_probability_app.command("run")
def ranker_probability_cmd(
    output: Path = typer.Option(..., help="Receipt/report path to publish (write-once)."),
    features: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="Existing gold/features.parquet."
    ),
    labels: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="Existing gold/labels.parquet."
    ),
    bronze_root: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        help="Canonical offline bronze directory to build gold from.",
    ),
    lake_root: Path | None = typer.Option(
        None, help="Derived silver/gold output directory (required with --bronze-root)."
    ),
    config: Path = typer.Option(Path("configs/base.yaml"), help="Config used for the gold build."),
    train_dates: int = typer.Option(252, help="Training dates."),
    cal_dates: int = typer.Option(63, help="Calibration dates."),
    test_dates: int = typer.Option(63, help="Test dates."),
    n_boot: int = typer.Option(2000, help="Bootstrap replicates."),
    feature_columns: list[str] | None = typer.Option(
        None, help="Feature column (repeatable). Default: PUBLIC_FEATURES."
    ),
    receipt_version: int = typer.Option(
        1, "--receipt-version", help="1 = strict-digest receipt (default), 2 = sealed receipt.v2."
    ),
) -> None:
    """Run the ranker-probability experiment and publish its receipt.

    Supply either a prebuilt gold pair (``--features`` + ``--labels``) or a
    bronze directory to build gold from (``--bronze-root`` + ``--lake-root``).
    The receipt carries the lane's own scope caveats — the data is exploratory
    and the holdout is not claimed unseen — so a passing gate is not a
    performance claim.
    """
    from quant_fund.cli.support import format_data_label
    from quant_fund.research.ranker_probability import main as ranker_main

    _check_receipt_version(receipt_version)
    if bronze_root is not None:
        if lake_root is None:
            raise typer.BadParameter("--bronze-root requires --lake-root")
        if features is not None or labels is not None:
            raise typer.BadParameter("--bronze-root excludes --features/--labels")
        source = str(bronze_root.name)
    else:
        if features is None or labels is None:
            raise typer.BadParameter(
                "pass both --features and --labels, or --bronze-root with --lake-root"
            )
        if lake_root is not None:
            raise typer.BadParameter("--lake-root requires --bronze-root")
        source = str(features.name)

    argv: list[str] = [
        "--output",
        str(output),
        "--config",
        str(config),
        "--train-dates",
        str(train_dates),
        "--cal-dates",
        str(cal_dates),
        "--test-dates",
        str(test_dates),
        "--n-boot",
        str(n_boot),
        "--receipt-version",
        str(receipt_version),
    ]
    if bronze_root is not None:
        argv += ["--bronze-root", str(bronze_root), "--lake-root", str(lake_root)]
    else:
        argv += ["--features", str(features), "--labels", str(labels)]
    for column in feature_columns or []:
        argv += ["--feature-columns", column]

    typer.echo(format_data_label(synthetic=False, data_source=source))
    # One implementation: delegate to the module's own main(argv). It signals
    # bad input via argparse's SystemExit(2), which is surfaced unchanged.
    try:
        code = ranker_main(argv)
    except SystemExit as exc:
        raise typer.Exit(code=int(exc.code or 1)) from None
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    raise typer.Exit(code=0 if code == 0 else int(code))
