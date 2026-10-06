"""Research command.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import typer

from .app import app
from .support import _cfg, format_data_label, format_fdr_families


def _harvest_p_values(findings: Iterable[Any]) -> list[float]:
    """Pull the ``stat == "p"`` values out of one harvest_findings result.

    Findings are dicts on the corpus-inference lane; attribute access is
    tolerated so the command survives a dataclass-shaped finding.
    """
    out: list[float] = []
    for finding in findings:
        if isinstance(finding, Mapping):
            stat, value = finding.get("stat"), finding.get("value")
        else:
            stat = getattr(finding, "stat", None)
            value = getattr(finding, "value", None)
        if stat == "p" and isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


@app.command()
def research(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    """Run Dipcatcher's proprietary research benches and write a labeled notebook."""
    from quant_fund.research.agent import format_p_value, run_research

    cfg = _cfg(config)
    nb = run_research(cfg)
    # Always print DATA_LABEL + FDR families (regression-tested).
    typer.echo(format_data_label(synthetic=bool(nb.synthetic), data_source=str(nb.data_source)))
    if nb.synthetic:
        typer.echo("SYNTHETIC")
    typer.echo(f"{nb.product} — {nb.firm}'s proprietary research lab {nb.version}")
    typer.echo(format_fdr_families(list(nb.hypotheses)))
    typer.echo(nb.disclaimer)
    for h in nb.hypotheses:
        typer.echo(f"{h.id}: {h.decision} (p={format_p_value(h.p_value)})")
    for r in nb.rankers:
        if str(r.get("name", "")).startswith("_"):
            dm_all = r.get("pairwise_dm_all") or []
            for d in dm_all:
                typer.echo(
                    f"DM {d.get('a')} vs {d.get('b')}: "
                    f"stat={d.get('statistic')} p={d.get('p_value')} preferred={d.get('preferred')}"
                )
            continue
        typer.echo(
            f"{r['name']}: IC={r['mean_ic']:.4f} RankIC={r.get('mean_rank_ic') or 0:.4f} "
            f"t={r['t_ic']:.2f} mono={r.get('decile_monotonicity') or 0:.3f}"
        )
    vol = nb.families.get("volatility", {})
    if vol:
        typer.echo(
            f"volatility QLIKE ewma={vol.get('qlike_ewma')} rolling={vol.get('qlike_rolling')}"
        )
    rl = nb.families.get("reinforcement", {})
    if rl:
        typer.echo(
            f"linucb reward={rl.get('mean_policy_reward')} "
            f"vs_uniform={rl.get('mean_advantage_vs_uniform')} "
            f"regret={rl.get('mean_regret_vs_oracle')}"
        )
    conf = nb.families.get("conformal", {})
    if conf:
        aci = conf.get("aci", {})
        raw = conf.get("gaussian_raw", {})
        wrap = conf.get("wrappee", "scaled_gaussian")
        typer.echo(
            f"conformal wrappee={wrap} "
            f"raw_cov={raw.get('coverage')} "
            f"cqr_raw_cov={conf.get('cqr_raw', {}).get('coverage')} "
            f"scaled_cov={conf.get('scaled', conf.get('scaled_gaussian', {})).get('coverage')} "
            f"cqr_cov={conf.get('cqr', {}).get('coverage')} "
            f"aci_cov={aci.get('coverage')} "
            f"aci_width={aci.get('mean_width')} "
            f"mondrian_cov={conf.get('mondrian_aci', {}).get('coverage')} "
            f"high_x={conf.get('mondrian_aci', {}).get('high_x_coverage')} "
            f"worst_x={conf.get('mondrian_aci', {}).get('worst_x_coverage')}"
        )
    ev = nb.families.get("evalues", {})
    if ev:
        typer.echo(
            f"evalues cov={ev.get('coverage')} e_final={ev.get('e_final')} "
            f"ever_cross={ev.get('ever_cross')}"
        )
    jp = nb.families.get("jackknife_plus", {})
    if jp:
        typer.echo(
            f"jackknife_plus cov={jp.get('coverage')} width={jp.get('mean_width')} "
            f"floor={jp.get('coverage_floor')}"
        )
    crc = nb.families.get("crc", {})
    if crc:
        typer.echo(
            f"crc wrappee={crc.get('wrappee')} risk={crc.get('risk')} "
            f"crc_stat={crc.get('crc_stat')} lambda={crc.get('lambda_hat')} "
            f"high_vol_bound={crc.get('high_vol_mean_bound')} "
            f"low_vol_bound={crc.get('low_vol_mean_bound')}"
        )
    wcqr = nb.families.get("weighted_conformal", {})
    if wcqr:
        typer.echo(
            f"weighted_conformal cov={wcqr.get('coverage')} "
            f"width={wcqr.get('mean_width')} "
            f"unweighted_cov={wcqr.get('unweighted_coverage')}"
        )
    caps = nb.families.get("interval_risk", {})
    if caps:
        typer.echo(
            f"interval_risk mean_cap={caps.get('mean_cap')} "
            f"frac_binding={caps.get('frac_binding')} "
            f"mean_width={caps.get('mean_width')}"
        )
    qb = nb.families.get("quantile_bandit", {})
    if qb:
        typer.echo(
            f"quantile_bandit reward={qb.get('mean_policy_reward')} "
            f"vs_uniform={qb.get('mean_advantage_vs_uniform')} "
            f"regret={qb.get('mean_regret_vs_oracle')}"
        )
    ns = nb.families.get("northset", {})
    if ns:
        typer.echo(
            f"northset ohlc_ok={ns.get('ohlc_identity_rate')} "
            f"book_ok={ns.get('book_uncrossed_rate')} "
            f"session_ok={ns.get('session_reconstructs_daily_rate')} "
            f"imb_ic={ns.get('imbalance_top_mean_ic')} "
            f"ofi_ic={ns.get('ofi_mean_ic')} "
            f"kyle_lambda={ns.get('kyle_lambda')} "
            f"park_qlike={ns.get('parkinson_qlike_vs_cc')} "
            f"gk_qlike={ns.get('garman_klass_qlike_vs_cc')} "
            f"cs_spread={ns.get('corwin_schultz_spread')} "
            f"chain_ok={ns.get('session_chain_rate')} "
            f"jump={ns.get('session_mean_jump_ratio')} "
            f"vpin={ns.get('vpin_mean')} "
            f"sweep_rate={ns.get('sweep_any_rate')} "
            f"sweep_rej_ic={ns.get('sweep_reject_signed_mean_ic')} "
            f"sweep_rej_event={ns.get('sweep_reject_event_mean_bps')}bps "
            f"sweep_follow_event={ns.get('sweep_follow_event_mean_bps')}bps "
            f"sweep_follow_costed={ns.get('sweep_follow_cost_adjusted_mean_bps')}bps "
            f"mean_session_spread_bps_mean={ns.get('mean_session_spread_bps_mean')} "
            f"mean_session_close_spread_bps={ns.get('mean_session_close_spread_bps')} "
            f"mean_session_close_imbalance={ns.get('mean_session_close_imbalance')} "
            f"mean_session_close_micro_bps={ns.get('mean_session_close_micro_bps')} "
            f"mean_session_close_mid={ns.get('mean_session_close_mid')} "
            f"mean_session_close_bid_depth={ns.get('mean_session_close_bid_depth')} "
            f"mean_session_close_ask_depth={ns.get('mean_session_close_ask_depth')} "
            f"mean_session_imbalance_std={ns.get('mean_session_imbalance_std')} "
            f"mean_session_imbalance_mean={ns.get('mean_session_imbalance_mean')} "
            f"session_ofi_sum_mean={ns.get('session_ofi_sum_mean')} "
            f"mean_session_ofi_abs_sum={ns.get('mean_session_ofi_abs_sum')} "
            f"session_book_vpin_mean={ns.get('session_book_vpin_mean')} "
            f"mean_session_book_snaps={ns.get('mean_session_book_snaps')}"
        )
    typer.echo(f"json={nb.artifacts.get('json')}")
    typer.echo(f"markdown={nb.artifacts.get('markdown')}")


@app.command("execution-sensitivity")
def execution_sensitivity_cmd(
    config: Path = typer.Option(Path("configs/backtest.yaml")),
    max_dates: int | None = typer.Option(
        None,
        help=(
            "Use only the first N decision dates present on both the feature "
            "panel and the causal gold panel."
        ),
    ),
    signal_latencies: str = typer.Option(
        "0,1", help="Comma-separated signal-to-order delays in bars."
    ),
    exchange_latencies: str = typer.Option(
        "0,1", help="Comma-separated order-to-exchange delays in bars."
    ),
    impact: str = typer.Option("1,2", help="Comma-separated square-root impact multipliers."),
    seed: int = typer.Option(0, help="Deterministic simulator seed."),
    initial_nav: float = typer.Option(1_000_000.0),
) -> None:
    """Latency/impact grid for one strategy. Execution diagnostic, not a live P&L claim."""
    import polars as pl

    from quant_fund.backtest.event_sim import execution_sensitivity, format_sensitivity_table
    from quant_fund.features.engine import build_features
    from quant_fund.pipeline.dataset import ensure_silver
    from quant_fund.pipeline.forecast import build_causal_weight_panel, decision_dates

    def _ints(raw: str) -> tuple[int, ...]:
        parts = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
        if not parts:
            raise typer.BadParameter("expected a comma-separated list of bar counts")
        return parts

    def _floats(raw: str) -> tuple[float, ...]:
        parts = tuple(float(piece.strip()) for piece in raw.split(",") if piece.strip())
        if not parts:
            raise typer.BadParameter("expected a comma-separated list of multipliers")
        return parts

    cfg = _cfg(config)
    bars = ensure_silver(cfg)
    feat = build_features(bars, cfg)
    # Gold drops warmup bars (history / label horizon). optimize_asof refuses
    # a decision date with no panel row, so the grid uses the overlap only.
    dates = decision_dates(cfg, feat["event_time"].unique().sort().to_list())
    if len(dates) < 2:
        raise typer.BadParameter(
            "need at least 2 decision dates on both the feature panel and the causal gold panel"
        )
    if max_dates is not None:
        if max_dates < 2:
            raise typer.BadParameter("--max-dates must be at least 2")
        dates = dates[:max_dates]
    feat = feat.filter(pl.col("event_time").is_in(dates))
    weights = build_causal_weight_panel(cfg, dates)
    report = execution_sensitivity(
        feat,
        weights,
        cfg,
        signal_latencies=_ints(signal_latencies),
        exchange_latencies=_ints(exchange_latencies),
        impact_multipliers=_floats(impact),
        initial_nav=initial_nav,
        seed=seed,
    )
    rows = report.get("rows")
    source = cfg.data.source
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        source = str(rows[0].get("source", source))
    synthetic = source.upper() == "SYNTHETIC" or "synthetic" in source.lower()
    typer.echo(format_data_label(synthetic=synthetic, data_source=source))
    if synthetic:
        typer.echo("SYNTHETIC")
    typer.echo(format_sensitivity_table(report))


@app.command("verify-identities")
def verify_identities(
    out: Path = typer.Option(
        Path("data/metadata/research/identity_sweep.json"), "--out", help="Receipt JSON path."
    ),
    trials: int = typer.Option(8, "--trials", help="Seeded SYNTHETIC draws per identity."),
    seed: int = typer.Option(7, "--seed", help="Base seed for the synthetic generators."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = identity_sweep v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Prove catalog/northset microstructure identities on SYNTHETIC draws.

    Prints the identity/residual table and writes an immutable receipt.
    Exits non-zero when any identity's max-abs residual exceeds its
    tolerance — fail-closed, CI gate candidate.
    """
    from quant_fund.research.identity_sweep import (
        format_identity_table,
        run_identity_sweep,
        write_identity_receipt,
    )

    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    receipt = run_identity_sweep(n_trials=int(trials), seed=int(seed))
    write_identity_receipt(out, receipt, receipt_version=receipt_version)
    typer.echo("SYNTHETIC")
    typer.echo(format_identity_table(receipt))
    typer.echo(f"receipt={out}")
    raise typer.Exit(code=0 if receipt["all_passed"] else 1)


@app.command()
def fleet(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = fleet_eval.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Run the SYNTHETIC distribution-challenger fleet and write a receipt.

    Proper scores only (pinball/CRPS/PIT-KS/coverage) on labeled synthetic
    shards — correctness evidence, never market or live-P&L claims.
    """
    from quant_fund.research.fleet_eval import (
        fleet_head_factories,
        resolve_shard_generators,
        run_distribution_fleet,
        write_fleet_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
        resolved_shards = resolve_shard_generators(None if shards is None else shards.split(","))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    frame, receipt = run_distribution_fleet(
        factories,
        resolved_shards,
        n_train=n_train,
        n_eval=n_eval,
        seed=base_seed,
        taus=cfg.quantiles.levels,
    )
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_fleet_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    typer.echo(frame)
    typer.echo(f"receipt={path}")


@app.command("calibration-eval")
def calibration_eval_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    pit_bins: int = typer.Option(10, help="PIT histogram bins (5-50)."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = calibration_eval.v1 (default), "
        "2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Run the SYNTHETIC distribution-fleet calibration lane and write a receipt.

    PIT histograms, central-interval coverage, and reliability/calibration
    slopes on the same origins as ``fleet`` — calibration diagnostics only,
    never market or live-P&L claims.
    """
    from quant_fund.research.calibration_eval import (
        run_calibration_eval,
        write_calibration_receipt,
    )
    from quant_fund.research.fleet_eval import (
        fleet_head_factories,
        resolve_shard_generators,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
        resolved_shards = resolve_shard_generators(None if shards is None else shards.split(","))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    frame, receipt = run_calibration_eval(
        factories,
        resolved_shards,
        n_train=n_train,
        n_eval=n_eval,
        seed=base_seed,
        taus=cfg.quantiles.levels,
        pit_bins=pit_bins,
    )
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_calibration_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(frame)
    typer.echo(f"receipt={path}")


@app.command("verify-receipt")
def verify_receipt_cmd(
    path: Path = typer.Argument(..., help="Receipt JSON file to verify."),
) -> None:
    """Verify a sealed receipt: structure plus hash consistency.

    ``receipt.v2`` envelopes are validated against the pydantic schema and
    their sealed digest, environment fingerprint, code-map digest, and (for
    known kinds) dataset/params digests are re-derived. Older v1 receipts get
    a ``receipt_sha256`` seal check (canonical or strict JSON convention);
    ``fleet_eval.v1`` payloads get their writer's contract too. Exits non-zero
    on any violation — fail-closed.
    """
    import json

    from quant_fund.research.receipt_v2 import verify_receipt_file

    result = verify_receipt_file(path)
    typer.echo(json.dumps(result, indent=2))
    raise typer.Exit(code=0 if result["valid"] else 1)


@app.command("verify-all")
def verify_all_cmd(
    receipts_dir: Path = typer.Option(Path("receipts"), help="Receipt directory to audit."),
    check_index: bool = typer.Option(
        True, help="Byte-compare docs/evidence/index.md against a fresh regen."
    ),
    write: bool = typer.Option(True, help="Write the sealed evidence_audit receipt."),
    out_dir: Path = typer.Option(Path("receipts"), help="Audit receipt output dir."),
) -> None:
    """Chain-of-custody audit over the whole receipts directory.

    Per-file verification via ``verify_receipt_file`` (v2 deep verification,
    v1 seal check); set-level checks a single-file verifier cannot express:
    filename↔digest binding, duplicate seals across files, unsealed-legacy
    accounting, and evidence-index staleness. Writes a sealed
    ``evidence_audit_<digest16>.json`` receipt. Exits non-zero on any hard
    finding — sealed-invalid, unparseable, filename mismatch, duplicate seal,
    or stale index.
    """
    from quant_fund.research.evidence_audit import (
        format_evidence_audit_table,
        run_evidence_audit,
        write_evidence_audit_receipt,
    )

    try:
        rows, receipt = run_evidence_audit(receipts_dir, check_index=check_index, root=Path.cwd())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_evidence_audit_table(rows))
    payload = receipt["payload"]
    typer.echo(
        f"files={payload['n_files']} sealed={payload['n_sealed']} "
        f"unsealed={payload['n_unsealed']} index_fresh={payload['index_fresh']}"
    )
    if payload["findings"]:
        typer.echo("findings: " + "; ".join(payload["findings"]))
    if write:
        path = write_evidence_audit_receipt(receipt, out_dir)
        typer.echo(f"receipt={path}")
    raise typer.Exit(code=0 if receipt["verdict"] == "pass" else 1)


@app.command("vol-bench")
def vol_bench(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None,
        help="Comma-separated vol model names (default: rv_roll, rv_ewma, har, "
        "realized_garch, dip_garch_t).",
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all SYNTHETIC vol shards)."
    ),
    horizons: str = typer.Option(
        "1,5", "--horizons", help="Comma-separated forecast horizons in bars."
    ),
    min_history: int = typer.Option(300, help="Leading fit bars per origin."),
    n_origins: int = typer.Option(24, help="Scored origins per shard/horizon (>=10)."),
    stride: int | None = typer.Option(None, help="Origin spacing in bars (default: max horizon)."),
    n_bars: int | None = typer.Option(
        None, help="Shard length (default: minimal for the origin schedule)."
    ),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = vol_bench.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Run the SYNTHETIC volatility bench and write a sealed receipt.

    QLIKE + MSE on next-bar/h-step cumulative realized variance for HAR,
    realized-GARCH, dip_garch_t and RV baselines over seeded synthetic vol
    shards (GARCH clustering, rough vol, structural breaks). Proper scores
    only — correctness evidence, never market or live-P&L claims.
    """
    from quant_fund.research.vol_bench import (
        resolve_vol_models,
        resolve_vol_shard_generators,
        run_vol_bench,
        write_vol_bench_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        horizon_set = tuple(int(h.strip()) for h in horizons.split(",") if h.strip())
        forecasters = resolve_vol_models(None if models is None else models.split(","))
        resolved_shards = resolve_vol_shard_generators(
            None if shards is None else shards.split(",")
        )
        frame, receipt = run_vol_bench(
            forecasters,
            resolved_shards,
            horizons=horizon_set,
            min_history=int(min_history),
            n_origins=int(n_origins),
            stride=None if stride is None else int(stride),
            n_bars=None if n_bars is None else int(n_bars),
            seed=int(base_seed),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_vol_bench_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    if receipt["data_label"] == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(frame)
    typer.echo(f"receipt={path}")


@app.command()
def rankic(
    panels: str | None = typer.Option(
        None, help="Comma-separated panel names (default: all synthetic panels)."
    ),
    challengers: str | None = typer.Option(
        None, help="Comma-separated challenger names (default: all transforms)."
    ),
    n_assets: int = typer.Option(32, help="Assets per date."),
    n_dates: int = typer.Option(96, help="Panel length in dates."),
    horizons: str = typer.Option("1,5,20", help="Comma-separated forward horizons."),
    seed: int = typer.Option(11, help="Base seed for the synthetic panels."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = cross_sectional_rankic.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Cross-sectional rank-IC bench on SYNTHETIC planted-signal panels (P3.4).

    Per-date Spearman rank-IC between each challenger transform and h-step
    forward returns, summarized with a Newey-West mean-IC t-stat. Proper-score
    framing only — never a P&L or live-trading claim.
    """
    from quant_fund.research.cross_sectional import (
        format_rankic_table,
        resolve_panels,
        run_cross_sectional_bench,
        write_rankic_receipt,
    )

    def _names(raw: str | None) -> list[str] | None:
        if raw is None:
            return None
        return [piece.strip() for piece in raw.split(",") if piece.strip()]

    try:
        resolved = resolve_panels(_names(panels))
        horizon_tuple = tuple(int(piece.strip()) for piece in horizons.split(",") if piece.strip())
        if not horizon_tuple:
            raise typer.BadParameter("--horizons must be nonempty")
        frame, receipt = run_cross_sectional_bench(
            resolved,
            challengers=_names(challengers),
            horizons=horizon_tuple,
            n_assets=n_assets,
            n_dates=n_dates,
            seed=seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_rankic_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    if receipt["data_label"] == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(format_rankic_table(frame))
    typer.echo(f"receipt={path}")


@app.command()
def capacity(
    books: str | None = typer.Option(
        None, help="Comma-separated synthetic book names (default: all)."
    ),
    n_dates: int = typer.Option(126, help="Dates per synthetic book."),
    n_names: int = typer.Option(32, help="Names per synthetic book."),
    seed: int = typer.Option(11, help="Base seed."),
    aum_grid: str = typer.Option(
        "1e6,1e7,5e7,1e8,5e8,1e9", help="Comma-separated AUM levels in dollars."
    ),
    participation_cap: float = typer.Option(
        0.10, help="Max share of dollar ADV a rebalance may consume per name-day."
    ),
    impact_coeff: float = typer.Option(0.1, help="Square-root impact coefficient."),
    dev: bool = typer.Option(False, "--dev", help="Acknowledge dev-only use; required to run."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = capacity_overlay.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """P5.6 participation-capacity bench on SYNTHETIC books (dev-only).

    Feasibility fractions, days-to-trade, and sqrt-impact cost in bps per
    book x AUM cell. Sealed `capacity_overlay_eval` receipt. Size bounds
    only — never a market or live-P&L claim.
    """
    if not dev:
        raise typer.BadParameter(
            "capacity is dev-only evidence tooling; pass --dev to acknowledge."
        )
    from quant_fund.research.capacity_overlay import (
        format_capacity_table,
        resolve_books,
        run_capacity_bench,
        write_capacity_receipt,
    )

    try:
        grid = tuple(float(x) for x in aum_grid.split(","))
        resolved = resolve_books(None if books is None else books.split(","))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    book_objs = [gen(n_dates, n_names, seed + i) for i, gen in enumerate(resolved.values())]
    frame, receipt = run_capacity_bench(
        book_objs,
        seed=seed,
        aum_grid=grid,
        participation_cap=participation_cap,
        impact_coeff=impact_coeff,
    )
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_capacity_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    if receipt["data_label"] == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(format_capacity_table(frame))
    typer.echo(f"receipt={path}")


@app.command()
def pairs(
    n_assets: int = typer.Option(8, help="Assets in the synthetic panel."),
    n_dates: int = typer.Option(600, help="Panel length in dates."),
    seed: int = typer.Option(0, help="Seed for the synthetic panel."),
    window: int = typer.Option(120, help="Trailing hedge-ratio window."),
    z_window: int = typer.Option(60, help="Trailing z-score window."),
    min_corr: float = typer.Option(0.5, help="Correlation pre-filter threshold."),
    alpha: float = typer.Option(0.05, help="BH rejection level."),
    entry: float = typer.Option(2.0, help="Z-score entry band."),
    exit_band: float = typer.Option(0.5, help="Z-score exit band."),
    hedge_method: str = typer.Option("ols", help="Hedge estimator: ols or kalman."),
    eval_horizon: int = typer.Option(1, help="Forward spread-change horizon."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Stat-arb pairs screen + PIT signal eval on a SYNTHETIC planted panel.

    Engle-Granger residual-ADF screen with a correlation pre-filter and
    BH/Bonferroni multiple-testing control, then a point-in-time z-score
    signal scored against forward spread changes. Proper-score framing
    only — detection truth and IC alignment, never a P&L claim.
    """
    from quant_fund.research.pairs import (
        format_pairs_table,
        run_pairs_eval,
        write_pairs_receipt,
    )

    try:
        frame, receipt = run_pairs_eval(
            seed=seed,
            n_assets=n_assets,
            n_dates=n_dates,
            window=window,
            z_window=z_window,
            min_corr=min_corr,
            alpha=alpha,
            entry=entry,
            exit=exit_band,
            hedge_method=hedge_method,
            eval_horizon=eval_horizon,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_pairs_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_pairs_table(frame))
    typer.echo(
        "planted_detected="
        f"{receipt['planted']['detected']} rank={receipt['planted']['rank_by_p_bh']} "
        f"spearman_ic={receipt['alignment']['spearman_ic']:+.4f}"
    )
    typer.echo(f"receipt={path}")


@app.command()
def race(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(256, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(128, help="Trailing eval rows, sliced into chunks."),
    n_chunks: int = typer.Option(8, help="Ordered eval chunks per shard (>=4, divides n_eval)."),
    alpha: float = typer.Option(0.05, help="Anytime-valid promotion level."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = fleet_race.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Sequential fleet elimination race on SYNTHETIC shards.

    Each head fits the leading slice and predicts the eval slice once;
    eval rows split into time-ordered chunks and two e-processes per head
    vs the chunk-0 incumbent give anytime-valid promotion/elimination
    verdicts (proper scores only — correctness evidence, never P&L).
    """
    from quant_fund.research.fleet_eval import (
        fleet_head_factories,
        resolve_shard_generators,
    )
    from quant_fund.research.fleet_race import fleet_race, write_race_receipt

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
        resolved = resolve_shard_generators(None if shards is None else shards.split(","))
        frame, receipt = fleet_race(
            factories,
            resolved,
            n_train=n_train,
            n_eval=n_eval,
            n_chunks=n_chunks,
            alpha=alpha,
            seed=base_seed,
            taus=cfg.quantiles.levels,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        path = write_race_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    typer.echo(
        frame.select(
            "shard", "model", "status", "promoted_at", "eliminated_at", "shard_winner", "verdict"
        )
    )
    typer.echo(f"receipt={path}")


@app.command("lane-power")
def lane_power(
    defects: str = typer.Option("0.0,0.1,0.25,0.5,1.0", help="Comma-separated defect sizes."),
    n_steps: int = typer.Option(400, help="Stream length per run."),
    n_seeds: int = typer.Option(20, help="Seeds per (lane, defect) cell."),
    alpha: float = typer.Option(0.05, help="Alarm threshold."),
    lanes: str | None = typer.Option(
        None, help="Comma-separated lane names (default: all available)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = lane_power.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Sequential power bench — measured alarm rate/time per monitor lane.

    Injects controlled defects into synthetic streams and records how fast
    each anytime-valid lane alarms; the defect=0 row bounds each lane's
    false-alarm rate by alpha. Lanes whose modules are unmerged show
    ``lane_missing`` on the receipt.
    """
    from quant_fund.research.lane_power import lane_power_bench, write_lane_power_receipt

    try:
        defect_grid = tuple(float(x) for x in defects.split(","))
    except ValueError as exc:
        raise typer.BadParameter(f"defects must be numeric: {exc}") from exc
    frame, receipt = lane_power_bench(
        defects=defect_grid,
        n_steps=n_steps,
        n_seeds=n_seeds,
        alpha=alpha,
        lanes=None if lanes is None else tuple(lanes.split(",")),
    )
    try:
        path = write_lane_power_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"lanes_ok={receipt['n_lanes_ok']} rows={frame.height} "
        f"null_alarm_rate={receipt['null_alarm_rate']}"
    )
    typer.echo(f"receipt={path}")


@app.command("suite-health")
def suite_health_cmd(
    receipts_dir: Path = typer.Option(
        Path("receipts"), help="Directory of committed receipts to re-verify."
    ),
    alpha: float = typer.Option(0.05, help="Pooled-evidence alarm threshold."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = suite_health.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit 1 unless every failing receipt is a byte-pinned legacy "
        "unsealed artifact (CI evidence-audit gate).",
    ),
) -> None:
    """Re-verify every receipt in a directory + pool evidence → sealed summary.

    One command audits the whole evidence trail: each file gets a fresh
    verify-receipt pass (seal + kind contract), harvestable p-values /
    e-values are pooled under arbitrary dependence, and a corrupt artifact
    withholds the pooled claim — never asserted over partial evidence.
    """
    from quant_fund.research.suite_health import suite_health, write_suite_health_receipt

    try:
        frame, receipt = suite_health(receipts_dir, alpha=alpha)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        path = write_suite_health_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=receipt["data_label"],
        )
    )
    typer.echo(
        f"receipts={receipt['n_receipts']} ok={receipt['n_ok']} "
        f"failed={receipt['n_failed']} pooled_evalue={receipt['pooled_evalue']}"
    )
    typer.echo(f"receipt={path}")
    if strict:
        import polars as pl

        from quant_fund.research.legacy_unsealed import (
            is_known_contract_legacy,
            is_known_unsealed,
        )
        from quant_fund.research.receipt_v2 import verify_receipt_file

        bad: list[str] = []
        for row in frame.filter(~pl.col("valid")).iter_rows(named=True):
            p = receipts_dir / str(row["file"])
            result = verify_receipt_file(p)
            if not (
                is_known_unsealed(p, result["errors"])
                or is_known_contract_legacy(p, result["errors"])
            ):
                bad.append(f"{row['file']}: {result['errors']}")
        if bad:
            typer.echo("STRICT FAILURE — unverifiable receipts:", err=True)
            for line in bad:
                typer.echo(f"  {line}", err=True)
            raise typer.Exit(code=1)


@app.command()
def mcs(
    streams: Path = typer.Argument(
        ...,
        help="Loss streams: JSON {head: [per-origin losses]} or parquet/csv "
        "(numeric column per head). Lower loss = better head.",
    ),
    alpha: float = typer.Option(0.05, help="Coverage failure level."),
    lam: float = typer.Option(0.5, help="Betting fraction per pair-process."),
    data_label: str = typer.Option("UNKNOWN", help="Provenance label stamped on the receipt."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = mcs_seq.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Sequential model confidence set over per-origin proper losses.

    Maintains a survivor set containing an optimal head with probability
    >= 1 - alpha *uniformly over time* (Ville + union bound over pairwise
    e-processes; permanent elimination). Writes a sealed mcs_seq.v1
    receipt — proper-score evidence only.
    """
    from quant_fund.research.mcs_seq import load_loss_streams, mcs_report, write_mcs_receipt

    try:
        loss_streams = load_loss_streams(streams)
        receipt = mcs_report(loss_streams, alpha=alpha, lam=lam, data_label=data_label)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        path = write_mcs_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    typer.echo(f"survivors={receipt['survivors']}")
    typer.echo(f"eliminated={receipt['eliminated']}")
    typer.echo(f"champion={receipt['champion']}")
    typer.echo(f"receipt={path}")


@app.command("serial-watch")
def serial_watch_cmd(
    pits: Path = typer.Argument(..., help="JSON file: a list of PITs in (0,1), or {name: [pits]}."),
    n_lags: int = typer.Option(5, help="Max lag for the sign-product families."),
    alpha: float = typer.Option(0.05, help="Per-family claim level."),
    lam: float = typer.Option(0.5, help="Bet cap λ ∈ (0,1)."),
    data_label: str = typer.Option("UNKNOWN", help="Provenance label stamped on each receipt."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = serial_watch.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Anytime-valid PIT serial-independence audit; sealed receipt per stream.

    Proper-scores evidence only — reports the per-lag family claim and the
    pooled e-value claim separately (never one merged flag; the joint claim
    boundary is per-family level α AND pooled level α, not one shared level).
    """
    import json

    from quant_fund.research.serial_watch import serial_report, write_serial_receipt

    try:
        raw = json.loads(pits.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"cannot read PIT file {pits}: {exc}") from exc
    streams: dict[str, object] = (
        {"stream": raw}
        if isinstance(raw, list)
        else {str(k): v for k, v in raw.items()}
        if isinstance(raw, dict)
        else {}
    )
    if not streams:
        raise typer.BadParameter("PIT file must be a JSON list or {name: [pits]} mapping")
    try:
        for name, stream in streams.items():
            receipt = serial_report(
                stream, n_lags=n_lags, alpha=alpha, lam=lam, data_label=data_label
            )
            path = write_serial_receipt(receipt, out_dir, receipt_version=receipt_version)
            typer.echo(
                format_data_label(synthetic=data_label == "SYNTHETIC", data_source=data_label)
            )
            typer.echo(
                f"{name}: alarmed_lags={receipt['alarmed_lags']} "
                f"(per-family claim, level {alpha}) pooled_evalue={receipt['pooled_evalue']:.4g} "
                f"pooled_alarmed={receipt['pooled_alarmed']} (separate pooled claim, level {alpha})"
            )
            typer.echo(f"receipt={path}")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("cost-calibration")
def cost_calibration(
    half_spread_bps: float = typer.Option(1.0, help="Flat half-spread floor in bps."),
    lookback: int = typer.Option(20, help="Trailing OHLC window for estimators."),
    n_dates: int = typer.Option(40, help="Dates in the SYNTHETIC panel."),
    n_names: int = typer.Option(4, help="Names in the SYNTHETIC panel."),
    seed: int = typer.Option(7, help="Panel seed."),
    planted_rel_spread: float = typer.Option(
        0.002, help="Planted high-low relative full spread for the SYNTHETIC book."
    ),
    dev: bool = typer.Option(False, "--dev", help="Acknowledge dev-only use; required to run."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    report_path: Path = typer.Option(
        Path("reports/cost_calibration_flat_vs_ohlc.md"),
        help="Markdown report path (flat vs calibrated trial table).",
    ),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = cost_calibration.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Flat vs OHLC-calibrated cost trials (dev-only SYNTHETIC diagnostic).

    Matched books under flat half-spread and Corwin–Schultz / Abdi–Ranaldo /
    Roll. Reports decomposed costs only — never Sharpe or live P&L.
    """
    if not dev:
        raise typer.BadParameter(
            "cost-calibration is dev-only evidence tooling; pass --dev to acknowledge."
        )
    from quant_fund.research.cost_calibration import (
        format_cost_calibration_table,
        run_cost_calibration_trials,
        write_cost_calibration_receipt,
        write_cost_calibration_report,
    )

    frame, receipt = run_cost_calibration_trials(
        half_spread_bps=half_spread_bps,
        lookback=lookback,
        n_dates=n_dates,
        n_names=n_names,
        seed=seed,
        planted_rel_spread=planted_rel_spread,
    )
    path = write_cost_calibration_receipt(receipt, out_dir, receipt_version=receipt_version)
    report = write_cost_calibration_report(frame, receipt, report_path)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_cost_calibration_table(frame))
    typer.echo(f"receipt={path}")
    typer.echo(f"report={report}")


@app.command()
def verdict(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    alpha: float = typer.Option(0.05, help="Confidence level for the verdict lanes."),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for winner's-curse."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = honest_verdict.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Fleet tournament → composite honest verdict → sealed receipt.

    Runs every head on the synthetic shards, keeps the per-origin loss
    and PIT streams (not just aggregates), and asks `honest_verdict`
    whether the winner's claim holds: promotion e-process, winner's-curse
    correction, drift alarm, magnitude CS, PIT calibration, changepoint
    localization. Verdicts: confirmed / supported_with_caveats /
    not_supported / inconclusive — inconclusive is a valid answer, never
    forced into a binary.
    """
    from quant_fund.research.fleet_eval import fleet_head_factories
    from quant_fund.research.verdict_run import run_verdict, write_verdict_receipt

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    report, status = run_verdict(
        factories,
        None if shards is None else shards.split(","),
        n_train=n_train,
        n_eval=n_eval,
        seed=base_seed,
        alpha=alpha,
        n_boot=n_boot,
        taus=cfg.quantiles.levels,
    )
    path = write_verdict_receipt(report, out_dir, receipt_version=receipt_version)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(f"verdict={report['verdict']} winner={report.get('winner')}")
    excluded = report["run"]["excluded_heads"]
    if excluded:
        typer.echo(f"excluded_heads={','.join(excluded)}")
    for name, detail in sorted(report["components"].items()):
        typer.echo(f"  {name}: {json.dumps(detail)[:200]}")
    typer.echo(f"receipt={path}")


@app.command(name="fleet-monitor")
def monitor(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    alpha: float = typer.Option(0.05, help="Alarm threshold (anytime-valid)."),
    level: float = typer.Option(0.9, help="Central interval for coverage lane."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = monitor_run.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Fleet tournament → every anytime-valid monitor lane → sealed receipt.

    Re-runs the tournament keeping per-origin quantiles, then streams each
    (shard, head) cell through the monitor family: coverage breach rate,
    nested tail depth, PIT calibration, conformal exchangeability, and
    loss drift vs the fleet median. Lanes whose modules are not merged
    report lane_missing on the receipt rather than failing silently.
    """
    from quant_fund.research.fleet_eval import fleet_head_factories
    from quant_fund.research.monitor_run import monitor_fleet, write_monitor_receipt

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    frame, receipt = monitor_fleet(
        factories,
        None if shards is None else shards.split(","),
        n_train=n_train,
        n_eval=n_eval,
        seed=base_seed,
        alpha=alpha,
        level=level,
        taus=cfg.quantiles.levels,
    )
    try:
        path = write_monitor_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"rows={receipt['n_rows']} alarms={receipt['n_alarm_rows']} "
        f"lanes={sum(receipt['lanes_available'].values())}/5"
    )
    typer.echo(f"receipt={path}")


@app.command()
def corpus(
    receipts_dir: Path = typer.Option(
        Path("receipts"), "--receipts-dir", help="Directory of committed receipts to audit."
    ),
    q: float = typer.Option(0.05, "--q", help="BH-FDR level for the pooled corpus family."),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = corpus_inference.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Pool every committed receipt's claims into one BH-FDR family.

    Harvests the p/e-value findings across ``--receipts-dir`` and writes a
    sealed ``corpus_inference.v1`` receipt naming which claims survive the
    corpus-level FDR correction — the selection-bias check lifted to the
    whole evidence store. Correctness evidence, never a market or P&L claim.
    """
    try:
        from quant_fund.research.corpus_inference import corpus_audit, write_corpus_receipt
    except ImportError as exc:
        raise typer.BadParameter("requires corpus_inference (PR #382)") from exc

    try:
        receipt = corpus_audit(receipts_dir, q=q)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        path = write_corpus_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"corpus receipts={receipt.get('n_receipts')} "
        f"p_findings={receipt.get('n_p_findings')} "
        f"survivors={receipt.get('n_survivors')} "
        f"corpus_evalue={receipt.get('corpus_evalue')} "
        f"parse_errors={receipt.get('n_parse_errors')}"
    )
    typer.echo(f"receipt={path}")


@app.command("online-fdr")
def online_fdr_cmd(
    receipts_dir: Path = typer.Option(
        Path("receipts"), "--receipts-dir", help="Directory of committed receipts to replay."
    ),
    level: float = typer.Option(0.05, "--level", help="Target mFDR bound for the stream."),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    per_receipt: bool = typer.Option(
        False,
        "--per-receipt",
        help="Treat each receipt as ONE test (min harvested p); default tests per finding.",
    ),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = online_fdr.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Replay committed receipts through Foster–Stine alpha-investing.

    Receipts replay in mtime/name order; every harvested p-value is one
    test (or one per receipt under ``--per-receipt``) fed to ``OnlineFDR``,
    whose wealth budget bounds the stream mFDR at ``--level`` at every
    arrival. Writes a sealed ``online_fdr.v1`` receipt. Correctness
    evidence, never a market or P&L claim.
    """
    try:
        from quant_fund.research.corpus_inference import harvest_findings
    except ImportError as exc:
        raise typer.BadParameter("requires corpus_inference (PR #382)") from exc
    try:
        from quant_fund.research.online_fdr import OnlineFDR, write_online_fdr_receipt
    except ImportError as exc:
        raise typer.BadParameter("requires online_fdr (PR #383)") from exc
    import json

    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    root = Path(receipts_dir)
    if not root.is_dir():
        raise typer.BadParameter(f"receipts dir {root} does not exist")
    try:
        controller = OnlineFDR(level=level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    files = sorted(
        (p for p in root.glob("*.json") if p.is_file()),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    digests: dict[str, str] = {}
    skipped: list[str] = []
    n_p_findings = 0
    for path in files:
        try:
            raw = path.read_bytes()
            payload: object = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError):
            skipped.append(path.name)
            continue
        if not isinstance(payload, Mapping):
            skipped.append(path.name)
            continue
        digests[path.name] = hash_bytes(raw)
        ps = _harvest_p_values(harvest_findings(payload, path.name))
        n_p_findings += len(ps)
        if per_receipt:
            if ps:
                controller.update(min(ps))
        else:
            for p in ps:
                controller.update(p)

    receipt = {
        "kind": "online_fdr.v1",
        "schema": "online_fdr.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "generated_at_commit": git_revision(),
        "inputs_sha256": hash_bytes(
            canonical_json_bytes({"digests": digests, "level": level, "per_receipt": per_receipt})
        ),
        "params": {
            "level": level,
            "per_receipt": per_receipt,
            "receipts_dir": str(root),
        },
        "n_receipts": len(files),
        "n_skipped": len(skipped),
        "skipped_files": skipped,
        "n_p_findings": n_p_findings,
        **controller.stream_report(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = write_online_fdr_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"online_fdr tests={receipt['n_tests']} rejections={receipt['n_rejections']} "
        f"final_wealth={receipt['final_wealth']:.6g} level={receipt['level']} "
        f"skipped={receipt['n_skipped']}"
    )
    typer.echo(f"receipt={path}")


__all__ = [
    "capacity",
    "cost_calibration",
    "corpus",
    "execution_sensitivity_cmd",
    "fleet",
    "pairs",
    "lane_power",
    "mcs",
    "monitor",
    "verdict",
    "lane_power",
    "monitor",
    "serial_watch_cmd",
    "suite_health_cmd",
    "verdict",
    "lane_power",
    "mcs",
    "monitor",
    "suite_health_cmd",
    "verdict",
    "race",
    "race",
    "race",
    "race",
    "suite_health_cmd",
    "monitor",
    "online_fdr_cmd",
    "rankic",
    "research",
    "verdict",
    "verify_identities",
    "verify_all_cmd",
    "verify_receipt_cmd",
    "vol_bench",
]


@app.command("lattice")
def lattice_cmd(
    receipts_dir: Path = typer.Option(
        Path("receipts"), "--receipts-dir", help="Directory of receipts to lattice."
    ),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    float_rel_tol: float = typer.Option(
        1e-9, "--float-rel-tol", help="Relative tolerance for numeric-drift edges."
    ),
    head_sha: str | None = typer.Option(
        None, "--head-sha", help="Current HEAD sha for stale-code flags (default: auto)."
    ),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = receipt_lattice.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
    known_inconsistent: Path | None = typer.Option(
        None,
        "--known-inconsistent",
        help="JSON map of receipt filename -> sha256 whose byte-exact "
        "inconsistent claim groups are acknowledged (demo artifacts).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit nonzero iff the verdict is 'inconsistent' (stale/drift/known-pinned pass).",
    ),
) -> None:
    """Cross-receipt consistency lattice over a receipts directory.

    Edges receipts that claim the same inputs/dataset fingerprints and
    compares their shared claim paths: consistent / numeric_drift /
    inconsistent. Provenance claims only — no P&L.
    """
    from quant_fund.research.receipt_lattice import receipt_lattice, write_lattice_receipt
    from quant_fund.utils.reproducibility import git_revision

    root = Path(receipts_dir)
    if not root.is_dir():
        raise typer.BadParameter(f"receipts dir {root} does not exist")
    pins: dict[str, str] | None = None
    if known_inconsistent is not None:
        if not known_inconsistent.is_file():
            raise typer.BadParameter(f"known-inconsistent file {known_inconsistent} does not exist")
        try:
            raw_pins = json.loads(known_inconsistent.read_text())
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"known-inconsistent is not JSON: {exc}") from exc
        if not isinstance(raw_pins, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and len(v) == 64 for k, v in raw_pins.items()
        ):
            raise typer.BadParameter(
                "known-inconsistent must be a JSON object mapping filename -> 64-hex sha256"
            )
        pins = dict(raw_pins)
    receipt = receipt_lattice(
        root,
        head_sha=head_sha or git_revision(),
        float_rel_tol=float_rel_tol,
        known_inconsistent=pins,
    )
    try:
        path = write_lattice_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"lattice receipts={receipt['n_receipts']} groups={receipt['n_claim_groups']} "
        f"verdict={receipt['verdict']}"
    )
    typer.echo(f"receipt={path}")
    if strict and receipt["verdict"] == "inconsistent":
        raise typer.Exit(code=1)


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
    DM-eliminated set — selection diagnostics only, never a live-trading claim.
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
        f"heads={len(factories)} shards={len(generators)} horizons={','.join(map(str, horizon_list))}"
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
