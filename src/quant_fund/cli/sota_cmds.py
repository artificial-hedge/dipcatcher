"""SOTA and forward-evidence command groups.

Four research lanes were reachable only as ``python -m`` scripts or as direct
imports from the ``scripts/*_col.py`` collectors, so they were invisible to
``dipcatcher --help`` and had no typed operator surface:

- :mod:`quant_fund.research.prospective_sota` — pre-registered forward SOTA
  comparison with a hash-chained event journal (commitment / prepare / forecast
  / settle / interrupt / verify).
- :mod:`quant_fund.research.forward_shadow` — the frozen-spec shadow run, which
  pulls in :mod:`shadow_journal` (hash-chained sqlite) and
  :mod:`forward_evidence` (HAC power planning).
- :mod:`quant_fund.research.sota_protocol` — protocol freeze, dual-board
  scoring and the promotion decision.
- :mod:`quant_fund.research.sota_evidence` — bar validation for the paired
  effect-interval collectors.

Each command is a thin typed wrapper over the same library function the
``python -m`` entry point calls, so the two surfaces cannot drift. The
``python -m`` mains and their subprocess contract tests are untouched.

Honesty contract: these lanes print JSON evidence documents. Nothing here is a
live-trading claim; a promotion decision is a research verdict over proper
scores, and every command echoes the DATA_LABEL of the inputs it was given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

prospective_sota_app = typer.Typer(
    help="Pre-registered prospective SOTA comparison over a hash-chained run journal."
)
forward_shadow_app = typer.Typer(
    help="Frozen-spec forward shadow run: freeze, decide, settle, reconcile. Simulation only."
)
sota_app = typer.Typer(help="SOTA protocol freeze, dual-board scoring, and promotion decisions.")


# ``cli.support`` imports ``cli._app``, and ``cli._app`` mounts this module, so
# a module-level import of the shared helpers here would be circular. Lazy
# re-exports break it the same way ``cli.main`` does for ``load_config``.
def format_data_label(*, synthetic: bool, data_source: str) -> str:
    """Lazy re-export. Importing the CLI must not re-enter ``cli._app``."""
    from quant_fund.cli.support import format_data_label as _impl

    return _impl(synthetic=synthetic, data_source=data_source)


def _cfg(config: Path) -> Any:
    """Lazy re-export. Importing the CLI must not load config."""
    from quant_fund.cli.support import _cfg as _impl

    return _impl(config)


def _read_json(path: Path) -> Any:
    """Load a JSON evidence document, rejecting duplicate keys.

    Mirrors ``prospective_sota._read`` — a duplicated key in a protocol or
    packet is an ambiguity, not a value, so it must fail rather than resolve.
    """
    from quant_fund.research.prospective_sota import _read

    return _read(Path(path))


def _emit(result: Any, output: Path | None) -> None:
    """Print one JSON evidence document, or write it and echo the path."""
    text = json.dumps(result, indent=2, allow_nan=False, default=str)
    if output is None:
        typer.echo(text)
        return
    from quant_fund.utils.atomicio import atomic_write_text

    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, text + "\n")
    typer.echo(f"written={output}")


# --------------------------------------------------------------------------- #
# prospective_sota
# --------------------------------------------------------------------------- #


@prospective_sota_app.command("commitment")
def prospective_commitment_cmd(
    protocol: Path = typer.Argument(..., exists=True, dir_okay=False, help="Protocol JSON."),
    output: Path | None = typer.Option(None, help="Write the commitment here instead of stdout."),
) -> None:
    """Bind a protocol: commitment, protocol and source digests, runtime.

    Run this *before* any forecast exists — the commitment digest is what makes
    a later settlement a test of a pre-registered claim rather than a post-hoc
    selection.
    """
    from quant_fund.research.prospective_sota import commitment

    try:
        result = commitment(_read_json(protocol))
    except (ValueError, KeyError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=protocol.name))
    _emit(result, output)


@prospective_sota_app.command("prepare")
def prospective_prepare_cmd(
    run_dir: Path = typer.Argument(..., help="Run journal directory (created if absent)."),
    protocol: Path = typer.Argument(..., exists=True, dir_okay=False, help="Protocol JSON."),
    anchor: Path = typer.Argument(..., exists=True, dir_okay=False, help="Anchor record JSON."),
    output: Path | None = typer.Option(None, help="Write the manifest here instead of stdout."),
) -> None:
    """Open a run journal against a committed protocol and an external anchor.

    Fails closed if the current time is not before the protocol's first origin:
    a prospective run cannot be prepared after the fact.
    """
    from quant_fund.research.prospective_sota import prepare

    try:
        result = prepare(run_dir, _read_json(protocol), _read_json(anchor))
    except (ValueError, KeyError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=protocol.name))
    _emit(result, output)


@prospective_sota_app.command("forecast")
def prospective_forecast_cmd(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Run journal directory."
    ),
    packet: Path = typer.Argument(..., exists=True, dir_okay=False, help="Forecast packet JSON."),
    output: Path | None = typer.Option(None, help="Write the event here instead of stdout."),
) -> None:
    """Append a forecast event to the journal (recorded before any outcome)."""
    from quant_fund.research.prospective_sota import forecast

    try:
        result = forecast(run_dir, _read_json(packet))
    except (ValueError, KeyError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit(result, output)


@prospective_sota_app.command("settle")
def prospective_settle_cmd(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Run journal directory."
    ),
    packet: Path = typer.Argument(..., exists=True, dir_okay=False, help="Settlement packet JSON."),
    output: Path | None = typer.Option(None, help="Write the event here instead of stdout."),
) -> None:
    """Append a settlement event, scoring the recorded forecasts against outcomes."""
    from quant_fund.research.prospective_sota import settle

    try:
        result = settle(run_dir, _read_json(packet))
    except (ValueError, KeyError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit(result, output)


@prospective_sota_app.command("interrupt")
def prospective_interrupt_cmd(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Run journal directory."
    ),
    reason: str = typer.Argument(..., help="Why the run stopped."),
    packet_sha256: str | None = typer.Option(None, help="Digest of the packet being abandoned."),
    output: Path | None = typer.Option(None, help="Write the event here instead of stdout."),
) -> None:
    """Record an interruption. Stopping a run is evidence, not a deletion."""
    from quant_fund.research.prospective_sota import interrupt

    if not reason.strip():
        raise typer.BadParameter("a non-empty reason is required")
    try:
        result = interrupt(run_dir, reason, packet_sha256)
    except (ValueError, KeyError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit(result, output)


@prospective_sota_app.command("verify")
def prospective_verify_cmd(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Run journal directory."
    ),
    output: Path | None = typer.Option(None, help="Write the verdict here instead of stdout."),
) -> None:
    """Re-derive the journal chain and report the pre-registered verdict.

    Exits non-zero when the chain does not verify — a broken chain is never
    reported as a pass.
    """
    from quant_fund.research.prospective_sota import verify

    try:
        result = verify(run_dir)
    except (ValueError, KeyError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit(result, output)
    if result.get("valid") is False:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# forward_shadow (+ shadow_journal, forward_evidence)
# --------------------------------------------------------------------------- #


@forward_shadow_app.command("freeze")
def forward_shadow_freeze_cmd(
    spec: Path = typer.Option(..., exists=True, dir_okay=False, help="Strategy spec JSON."),
    bootstrap: Path = typer.Option(..., exists=True, dir_okay=False, help="Bootstrap bars JSON."),
    run: Path = typer.Option(..., help="Shadow journal path to create."),
    output: Path | None = typer.Option(None, help="Write the manifest here instead of stdout."),
) -> None:
    """Freeze a strategy spec and open its hash-chained shadow journal.

    The spec digest is bound into the journal at creation, so a later decision
    cannot be attributed to a spec that was edited after the fact. See
    ``configs/forward_shadow.example.json``.
    """
    from quant_fund.research.forward_shadow import freeze

    try:
        result = freeze(json.loads(spec.read_text()), json.loads(bootstrap.read_text()), Path(run))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=spec.name))
    _emit(result, output)


def _record_event(operation: str, run: Path, payload: Path, output: Path | None) -> None:
    """Shared body for the decide / miss / settle journal events."""
    from quant_fund.research.forward_shadow import record

    try:
        event = record(Path(run), operation, json.loads(Path(payload).read_text()))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit(
        {key: event[key] for key in ("seq", "hash", "kind", "recorded_at") if key in event}, output
    )


@forward_shadow_app.command("decide")
def forward_shadow_decide_cmd(
    run: Path = typer.Option(..., exists=True, dir_okay=False, help="Shadow journal path."),
    input: Path = typer.Option(..., exists=True, dir_okay=False, help="Decision request JSON."),
    output: Path | None = typer.Option(None, help="Write the event here instead of stdout."),
) -> None:
    """Append a decision event to the shadow journal."""
    _record_event("decide", run, input, output)


@forward_shadow_app.command("miss")
def forward_shadow_miss_cmd(
    run: Path = typer.Option(..., exists=True, dir_okay=False, help="Shadow journal path."),
    input: Path = typer.Option(..., exists=True, dir_okay=False, help="Miss request JSON."),
    output: Path | None = typer.Option(None, help="Write the event here instead of stdout."),
) -> None:
    """Append a missed-opportunity event. Recording what was skipped is the point."""
    _record_event("miss", run, input, output)


@forward_shadow_app.command("settle")
def forward_shadow_settle_cmd(
    run: Path = typer.Option(..., exists=True, dir_okay=False, help="Shadow journal path."),
    input: Path = typer.Option(..., exists=True, dir_okay=False, help="Settlement request JSON."),
    output: Path | None = typer.Option(None, help="Write the event here instead of stdout."),
) -> None:
    """Append a settlement event to the shadow journal."""
    _record_event("settle", run, input, output)


@forward_shadow_app.command("reconcile")
def forward_shadow_reconcile_cmd(
    run: Path = typer.Option(..., exists=True, dir_okay=False, help="Shadow journal path."),
    repair: bool = typer.Option(False, help="Rewrite derived state from the event chain."),
    expected_head: str | None = typer.Option(None, help="Assert the journal head digest."),
    output: Path | None = typer.Option(None, help="Write the report here instead of stdout."),
) -> None:
    """Replay the journal and reconcile derived state against the chain.

    Exits non-zero when the chain is broken or the head does not match
    ``--expected-head`` — reconciliation is a gate, not a summary.
    """
    from quant_fund.research.forward_shadow import report

    try:
        result = report(Path(run), repair=repair, expected_head=expected_head)
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    summary = {key: value for key, value in result.items() if key not in {"events", "state"}}
    _emit(summary, output)
    if result.get("consistent") is False or result.get("ok") is False:
        raise typer.Exit(code=1)


@forward_shadow_app.command("dump")
def forward_shadow_dump_cmd(
    run: Path = typer.Argument(..., exists=True, dir_okay=False, help="Shadow journal path."),
    output: Path | None = typer.Option(None, help="Write the events here instead of stdout."),
) -> None:
    """Dump the hash-chained journal events (read-only audit view)."""
    from quant_fund.research.shadow_journal import connect, read_events

    connection = connect(Path(run))
    try:
        events = read_events(connection)
    finally:
        connection.close()
    _emit(events, output)


@forward_shadow_app.command("plan")
def forward_shadow_plan_cmd(
    calibration: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        help="JSON array of calibration net-difference observations.",
    ),
    effect_bps: float = typer.Option(..., help="Effect size to detect, in basis points."),
    lag: int = typer.Option(..., help="Bartlett HAC lag for the long-run variance."),
    alpha: float = typer.Option(0.05, help="One-sided significance level, in (0, 0.5)."),
    power: float = typer.Option(0.80, help="Target power, in (0.5, 1)."),
    output: Path | None = typer.Option(None, help="Write the plan here instead of stdout."),
) -> None:
    """Plan how many forward sessions an effect needs before it is measurable.

    Bartlett HAC long-run variance on a calibration sample, one-sided normal
    approximation. The plan states its own assumption and carries no power
    guarantee: it sizes an experiment, it does not validate a strategy.
    """
    from quant_fund.research.forward_evidence import evidence_plan

    try:
        series = json.loads(Path(calibration).read_text())
        if not isinstance(series, list):
            raise ValueError("calibration must be a JSON array of numbers")
        result = evidence_plan(
            [float(value) for value in series],
            effect_bps=effect_bps,
            lag=lag,
            alpha=alpha,
            power=power,
        )
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=Path(calibration).name))
    _emit(result, output)


# --------------------------------------------------------------------------- #
# sota_protocol / sota_evidence
# --------------------------------------------------------------------------- #


@sota_app.command("show")
def sota_show_cmd(
    protocol: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="SOTA protocol JSON/YAML."
    ),
    output: Path | None = typer.Option(None, help="Write the resolved protocol here."),
) -> None:
    """Resolve and fingerprint a SOTA protocol without scoring anything.

    Prints the protocol digest that a run would be bound to, plus the frozen
    fields — the cheap check to run before committing compute to ``sota run``.
    """
    from quant_fund.research.sota_protocol import load_sota_protocol, protocol_sha256

    try:
        resolved = load_sota_protocol(protocol)
        digest = protocol_sha256(resolved)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"protocol_sha256={digest}")
    _emit(resolved.model_dump(mode="json"), output)


@sota_app.command("run")
def sota_run_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    protocol: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="SOTA protocol JSON/YAML."
    ),
    frame: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Pre-built panel parquet (default: build from config).",
    ),
    no_torch: bool = typer.Option(False, help="Skip torch-gated heads."),
    no_path_rankic: bool = typer.Option(False, help="Skip the path RankIC board."),
    no_calibration: bool = typer.Option(False, help="Skip the calibration gate."),
    receipt_version: int = typer.Option(
        1, "--receipt-version", help="1 = lane receipt (default), 2 = sealed receipt.v2 envelope."
    ),
    output: Path | None = typer.Option(None, help="Write the evidence document here."),
) -> None:
    """Freeze a protocol, then score both boards and the calibration gate.

    Proper scores only. A promotion decision here is a research verdict over
    pre-registered gates — never a live-trading or market-evidence claim.
    """
    import polars as pl

    from quant_fund.research.sota_protocol import load_sota_protocol, run_sota_protocol

    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    cfg = _cfg(config)
    try:
        resolved = load_sota_protocol(protocol)
        panel = pl.read_parquet(frame) if frame is not None else None
        result = run_sota_protocol(
            cfg,
            resolved,
            panel,
            include_torch=not no_torch,
            include_path_rankic=not no_path_rankic,
            include_calibration=not no_calibration,
            receipt_version=receipt_version,
        )
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    synthetic = frame is None and str(cfg.data.source).lower() == "synthetic"
    typer.echo(
        format_data_label(
            synthetic=synthetic,
            data_source="SYNTHETIC"
            if synthetic
            else (frame.name if frame else str(cfg.data.source)),
        )
    )
    decision = result.get("promotion") or result.get("decision")
    if isinstance(decision, dict):
        typer.echo(f"promotion_decision={decision.get('decision')}")
    _emit(result, output)


@sota_app.command("validate-bars")
def sota_validate_bars_cmd(
    bars: Path = typer.Argument(..., exists=True, dir_okay=False, help="Bars parquet to validate."),
) -> None:
    """Validate a bar panel the paired-effect collectors would consume.

    This is the input gate the ``scripts/*_col.py`` collectors run in-process;
    exposed here so an operator can check a panel before spending compute on a
    SOTA evidence run. Exits non-zero on an unusable panel.
    """
    import polars as pl

    from quant_fund.research.sota_evidence import validate_bars

    try:
        times, interval_ns = validate_bars(pl.read_parquet(bars))
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=bars.name))
    typer.echo(f"bars={len(times)} bar_interval_ns={interval_ns}")
    if len(times) < 2:
        raise typer.Exit(code=1)


def _load_loss_panel(path: Path) -> tuple[Any, list[str]]:
    """Read a chronological loss panel: one numeric column per model.

    The inference functions in :mod:`quant_fund.research.sota_evidence` take an
    already-aligned ``(n_target_times, n_models)`` loss matrix, so the file
    convention is the panel itself — columns are models in file order, rows are
    consecutive target times. No index column is interpreted; ordering is the
    file's, which is why the lane requires the caller to have aligned it.
    """
    import numpy as np
    import polars as pl

    if not path.is_file():
        raise ValueError(f"loss panel not found: {path}")
    frame = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)
    numeric = [c for c in frame.columns if frame[c].dtype.is_numeric()]
    if len(numeric) < 2:
        raise ValueError(
            f"{path.name} needs at least 2 numeric loss columns (one per model), "
            f"found {len(numeric)}"
        )
    matrix = frame.select(numeric).cast(pl.Float64).to_numpy()
    if matrix.shape[0] < 2:
        raise ValueError(f"{path.name} needs at least 2 rows")
    if not np.isfinite(matrix).all():
        raise ValueError("loss panel must be finite — missing observations cannot be bootstrapped")
    if np.any(matrix < 0):
        raise ValueError("proper losses must be nonnegative")
    return matrix, numeric


def _resolve_targets(targets: str, model_names: list[str]) -> list[str]:
    names = [t.strip() for t in targets.split(",") if t.strip()]
    if not names:
        raise ValueError("--targets must name at least one model")
    unknown = sorted(set(names) - set(model_names))
    if unknown:
        raise ValueError(
            f"target(s) {', '.join(unknown)} are not columns of the loss panel; "
            f"available: {', '.join(model_names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("--targets must be unique")
    return names


@sota_app.command("effects")
def sota_effects_cmd(
    losses: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        help="Chronological loss panel: one numeric column per model.",
    ),
    targets: str = typer.Option(
        ..., help="Comma-separated model names to treat as the candidate targets."
    ),
    n_boot: int = typer.Option(2000, help="Stationary-bootstrap replicates (>=2)."),
    seed: int = typer.Option(7, help="Bootstrap seed."),
    block: float | None = typer.Option(
        None, help="Expected block length (default: median optimal block length across pairs)."
    ),
    alpha: float = typer.Option(0.05, help="Significance level, in (0, 1)."),
    output: Path | None = typer.Option(None, help="Write the intervals here instead of stdout."),
) -> None:
    """Paired CRPS effects with simultaneous bootstrap intervals.

    One stationary bootstrap draw resamples every comparison together, so the
    maximum standardized deviation gives intervals that hold over the whole
    declared target/comparator family — not just pointwise. Positive differences
    favor the comparator. These are approximate bootstrap intervals, never
    distribution-free bounds, and never a live-performance claim.
    """
    from quant_fund.research.sota_evidence import paired_effect_intervals

    try:
        matrix, model_names = _load_loss_panel(losses)
        target_names = _resolve_targets(targets, model_names)
        result = paired_effect_intervals(
            matrix,
            model_names,
            target_names,
            n_boot=n_boot,
            seed=seed,
            block=block,
            alpha=alpha,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=losses.name))
    typer.echo(f"models={len(model_names)} targets={len(target_names)} rows={matrix.shape[0]}")
    _emit(result, output)


@sota_app.command("block-sensitivity")
def sota_block_sensitivity_cmd(
    losses: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        help="Chronological loss panel: one numeric column per model.",
    ),
    targets: str = typer.Option(..., help="Comma-separated candidate target model names."),
    blocks: str = typer.Option(
        "1,5,10,20", help="Comma-separated expected block lengths. 1 is the iid diagnostic."
    ),
    n_boot: int = typer.Option(2000, help="Stationary-bootstrap replicates per block length."),
    seed: int = typer.Option(7, help="Bootstrap seed."),
    output: Path | None = typer.Option(None, help="Write the sensitivity table here."),
) -> None:
    """Show how much the inference depends on the prespecified block length.

    Block length 1 is the iid diagnostic; longer blocks retain serial
    dependence. The lengths are declared up front and every one is reported —
    they are not searched for a favorable p-value, which is the disclosure this
    command exists to make easy.
    """
    from quant_fund.research.sota_evidence import block_sensitivity

    try:
        block_values = tuple(float(b) for b in blocks.split(",") if b.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"--blocks must be comma-separated numbers: {exc}") from exc
    if not block_values or any(b <= 0 for b in block_values):
        raise typer.BadParameter("--blocks must be positive")
    try:
        matrix, model_names = _load_loss_panel(losses)
        target_names = _resolve_targets(targets, model_names)
        result = block_sensitivity(
            matrix,
            model_names,
            target_names,
            n_boot=n_boot,
            seed=seed,
            blocks=block_values,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=False, data_source=losses.name))
    typer.echo(
        f"models={len(model_names)} targets={len(target_names)} rows={matrix.shape[0]} "
        f"blocks={','.join(f'{b:g}' for b in block_values)}"
    )
    _emit(result, output)
