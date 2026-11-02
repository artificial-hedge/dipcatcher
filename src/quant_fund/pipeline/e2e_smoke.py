"""Deterministic offline ingest-to-report smoke.

SYNTHETIC correctness only. Two runs with the same seed must share a digest.
A NaN, an infinity, or an empty stage raises ``E2ESmokeError``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from quant_fund.backtest.engine import BacktestResult, run_backtest
from quant_fund.backtest.listing_exit import ListingExitError, schedule_listing_exits
from quant_fund.config.loader import load_config
from quant_fund.config.models import AppConfig
from quant_fund.data.ingest import ingest
from quant_fund.pipeline.dataset import build_gold, clear_panel_cache, ensure_silver
from quant_fund.pipeline.forecast import (
    build_causal_weight_panel,
    clear_forecast_caches,
    decision_dates,
    forecast_asof,
)
from quant_fund.pipeline.train import train_family
from quant_fund.portfolio.optimizer import component_risk
from quant_fund.portfolio.risk_gate import check_order
from quant_fund.reporting.report import write_report
from quant_fund.reporting.tearsheet import build_tearsheet, write_tearsheet_md
from quant_fund.risk.gates import GateSpec, apply_gate_stack
from quant_fund.risk.pyriskmgmt import portfolio_var_es
from quant_fund.schemas.errors import RiskGateRejected
from quant_fund.schemas.forecast import AssetForecast
from quant_fund.schemas.orders import Order, OrderSide
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes, hash_file
from quant_fund.utils.seeds import set_global_seed

_MAX_ASSETS = 12
_MAX_DAYS = 160
_SEED_LIMIT = 2**32
_PRETRADE_NAV = 1_000_000.0
_STAGE_NAMES = (
    "ingest",
    "features",
    "train",
    "forecast",
    "backtest",
    "portfolio",
    "risk",
    "report",
)


class E2ESmokeError(RuntimeError):
    """A smoke stage was empty, non-finite, or outside the offline contract."""


def default_config_path() -> Path:
    """Repo ``configs/e2e_smoke.yaml`` next to this source tree."""
    return Path(__file__).resolve().parents[3] / "configs" / "e2e_smoke.yaml"


def require_rows(stage: str, frame: pl.DataFrame) -> None:
    """Reject a frame with no rows."""
    if frame.height == 0:
        raise E2ESmokeError(f"{stage}: empty")


def require_no_nonfinite_floats(stage: str, frame: pl.DataFrame) -> None:
    """Reject NaN or inf in float columns. Nulls are allowed."""
    bad = [name for name, dtype in frame.schema.items() if _float_column_bad(frame, name, dtype)]
    if bad:
        shown = ", ".join(bad[:8])
        raise E2ESmokeError(f"{stage}: non-finite floats in {shown}")


def require_populated(stage: str, frame: pl.DataFrame, columns: tuple[str, ...]) -> None:
    """Reject a missing, empty, null, or non-finite named column."""
    require_rows(stage, frame)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise E2ESmokeError(f"{stage}: missing columns {missing}")
    for name in columns:
        column = frame.get_column(name)
        if column.null_count():
            raise E2ESmokeError(f"{stage}: nulls in {name}")
        if column.dtype in (pl.Float32, pl.Float64) and _series_nonfinite(column):
            raise E2ESmokeError(f"{stage}: non-finite {name}")


def require_finite(stage: str, value: object) -> None:
    """Walk a receipt value and raise on NaN, inf, or an unsupported type."""
    bad: list[str] = []
    _walk(stage, value, bad)
    if bad:
        extra = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
        raise E2ESmokeError(f"{stage}: non-finite at {', '.join(bad[:8])}{extra}")


def run_e2e_smoke(
    root: str | Path,
    *,
    config_path: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run ingest through report and return a digest-bearing receipt."""
    lake = Path(root)
    lake.mkdir(parents=True, exist_ok=True)
    chosen = None if config_path is None else Path(config_path)
    cfg = _load_smoke_config(lake, chosen, seed)
    stages = _execute(lake, cfg)
    return _receipt(cfg, stages)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: print the receipt JSON, or write a stage error to stderr."""
    parser = argparse.ArgumentParser(
        description="Deterministic SYNTHETIC ingest-to-report correctness smoke"
    )
    parser.add_argument("--root", required=True, help="Writable lake root for this run")
    parser.add_argument(
        "--config", default=None, help="Config YAML (default configs/e2e_smoke.yaml)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the synthetic seed")
    args = parser.parse_args(argv)
    try:
        receipt = run_e2e_smoke(args.root, config_path=args.config, seed=args.seed)
    except E2ESmokeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(canonical_json_bytes(receipt).decode("utf-8"))
    sys.stdout.write("\n")
    return 0


def _load_smoke_config(root: Path, config_path: Path | None, seed: int | None) -> AppConfig:
    cfg = load_config(config_path or default_config_path())
    data = cfg.data.model_copy(update={"root": root})
    train = cfg.train
    validation = cfg.validation
    if seed is not None:
        _require_seed(seed)
        data = data.model_copy(update={"synthetic_seed": seed})
        train = train.model_copy(update={"random_seed": seed})
        validation = validation.model_copy(update={"seed": seed})
    cfg = cfg.model_copy(update={"data": data, "train": train, "validation": validation})
    _reject_unbounded(cfg)
    return cfg


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < _SEED_LIMIT:
        raise E2ESmokeError("seed must be an int in [0, 2**32)")


def _reject_unbounded(cfg: AppConfig) -> None:
    if cfg.data.source != "synthetic":
        raise E2ESmokeError("e2e smoke refuses a non-synthetic source")
    if cfg.robinhood_plus.enabled:
        raise E2ESmokeError("e2e smoke refuses robinhood_plus")
    assets = int(cfg.data.synthetic_n_assets)
    days = int(cfg.data.synthetic_n_days)
    if assets > _MAX_ASSETS or days > _MAX_DAYS:
        raise E2ESmokeError(
            f"e2e smoke refuses panels above {_MAX_ASSETS} assets or {_MAX_DAYS} days"
        )


def _execute(root: Path, cfg: AppConfig) -> dict[str, Any]:
    set_global_seed(int(cfg.data.synthetic_seed))
    clear_panel_cache()
    clear_forecast_caches()
    db = root / "metadata" / "mlflow.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with _thread_limits(), _isolated_tracking(db):
        return _stages(root, cfg)


def _stages(root: Path, cfg: AppConfig) -> dict[str, Any]:
    ingest_stage, bars = _ingest(cfg)
    stages: dict[str, Any] = {
        "ingest": ingest_stage,
        "features": _features(cfg),
        "train": _train(cfg),
        "forecast": _forecast(cfg),
    }
    backtest_stage, weights, result = _backtest(cfg, bars)
    exits = int(backtest_stage["listing_exits"])
    stages["backtest"] = backtest_stage
    stages["portfolio"] = _portfolio(weights, bars, exits)
    stages["risk"] = _risk(cfg, result, weights, bars)
    _reject_empty_stages(stages)
    stages["report"] = _report(root, stages, result)
    _reject_empty_stages(stages)
    return stages


def _ingest(cfg: AppConfig) -> tuple[dict[str, Any], pl.DataFrame]:
    paths = ingest(cfg)
    bars = ensure_silver(cfg)
    require_populated("ingest", bars, ("open", "high", "low", "close", "volume"))
    names = bars.get_column("security_id").n_unique()
    if names < 2:
        raise E2ESmokeError("ingest: fewer than 2 names")
    if "source" not in bars.columns:
        raise E2ESmokeError("ingest: source column is missing")
    sources = {str(value) for value in bars.get_column("source").unique().to_list()}
    if sources != {"synthetic"}:
        raise E2ESmokeError("ingest: source is not synthetic")
    if not paths:
        raise E2ESmokeError("ingest: empty")
    stage = {"rows": int(bars.height), "names": int(names), "artifacts": len(paths)}
    return stage, bars


def _features(cfg: AppConfig) -> dict[str, Any]:
    feats, labs = build_gold(cfg)
    require_rows("features", feats)
    require_rows("features", labs)
    require_no_nonfinite_floats("features", feats)
    require_no_nonfinite_floats("features", labs)
    if "future_excess_return_1" not in labs.columns:
        raise E2ESmokeError("features: missing future_excess_return_1")
    observed = labs.get_column("future_excess_return_1").drop_nulls()
    if observed.len() < 1 or _series_nonfinite(observed):
        raise E2ESmokeError("features: future_excess_return_1 is empty or non-finite")
    return {
        "feature_rows": int(feats.height),
        "feature_cols": int(len(feats.columns)),
        "label_rows": int(labs.height),
    }


def _train(cfg: AppConfig) -> dict[str, Any]:
    result = train_family(cfg, "ranking", "ridge")
    if result.get("data_source") != "SYNTHETIC":
        raise E2ESmokeError("train: data_source is not SYNTHETIC")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise E2ESmokeError("train: metrics missing")
    stage: dict[str, Any] = {
        "data_source": "SYNTHETIC",
        "family": "ranking",
        "model": "ridge",
    }
    for key in ("mean_ic", "mean_rank_ic", "icir", "ic_tstat", "ic_pvalue"):
        stage[key] = _required_float("train", metrics, key)
    n_dates = _required_int("train", metrics, "n_ic_dates")
    if n_dates < 1:
        raise E2ESmokeError("train: n_ic_dates is empty")
    stage["n_ic_dates"] = n_dates
    return stage


def _forecast(cfg: AppConfig) -> dict[str, Any]:
    state = forecast_asof(cfg)
    notes = [str(note) for note in state.notes]
    if not any("SYNTHETIC" in note for note in notes):
        raise E2ESmokeError("forecast: missing SYNTHETIC label")
    if not state.forecasts:
        raise E2ESmokeError("forecast: empty")
    ordered = sorted(state.forecasts, key=lambda item: item.security_id)
    ids = [item.security_id for item in ordered]
    if len(set(ids)) != len(ids):
        raise E2ESmokeError("forecast: duplicate security_id")
    by_name = {item.security_id: _forecast_row(item) for item in ordered}
    stage: dict[str, Any] = {
        "n": len(ordered),
        "asof": state.asof.isoformat(),
        "names": ids,
        "by_name": by_name,
    }
    if notes and all(_path_free(note) for note in notes):
        stage["notes"] = sorted(notes)
    return stage


def _forecast_row(forecast: AssetForecast) -> dict[str, Any]:
    alphas = {
        key: _finite("forecast", key, _finite_number(value))
        for key, value in sorted(forecast.alpha.items())
    }
    if not alphas or forecast.interval_method is None or not forecast.interval_lo:
        raise E2ESmokeError(f"forecast: {forecast.security_id} has no finite alpha or interval")
    intervals: dict[str, dict[str, float]] = {}
    for key in sorted(set(forecast.interval_lo) | set(forecast.interval_hi)):
        lo = forecast.interval_lo.get(key)
        hi = forecast.interval_hi.get(key)
        if lo is None or hi is None:
            raise E2ESmokeError(f"forecast: {forecast.security_id} interval {key} is incomplete")
        intervals[key] = {
            "lo": _finite("forecast", key, _finite_number(lo)),
            "hi": _finite("forecast", key, _finite_number(hi)),
        }
    if not intervals:
        raise E2ESmokeError(f"forecast: {forecast.security_id} has no interval")
    return {"alpha": alphas, "intervals": intervals, "method": forecast.interval_method}


def _backtest(
    cfg: AppConfig, bars: pl.DataFrame
) -> tuple[dict[str, Any], pl.DataFrame, BacktestResult]:
    from quant_fund.features.engine import build_features

    feat = build_features(bars, cfg)
    require_rows("backtest", feat)
    dates = decision_dates(cfg, feat.get_column("event_time").unique().sort().to_list())
    if len(dates) < 2:
        raise E2ESmokeError("backtest: fewer than 2 decision dates")
    raw = build_causal_weight_panel(cfg, dates)
    require_rows("backtest", raw)
    require_no_nonfinite_floats("backtest", raw)
    try:
        weights = schedule_listing_exits(raw, bars, _read_master(cfg))
    except ListingExitError as exc:
        raise E2ESmokeError(f"portfolio: {exc}") from exc
    require_populated("backtest", weights, ("event_time", "security_id", "target_weight"))
    require_no_nonfinite_floats("backtest", weights)
    result = run_backtest(bars, weights, cfg, fast=False)
    if result.source_note != "SYNTHETIC":
        raise E2ESmokeError("backtest: source_note is not SYNTHETIC")
    if result.equity.height < 2:
        raise E2ESmokeError("backtest: equity is empty")
    require_populated("backtest", result.equity, ("nav",))
    require_rows("backtest", result.fills)
    require_no_nonfinite_floats("backtest", result.equity)
    require_no_nonfinite_floats("backtest", result.fills)
    stage = {
        "equity_rows": int(result.equity.height),
        "fill_rows": int(result.fills.height),
        "total_return": _required_float("backtest", result.metrics, "total_return"),
        "max_drawdown": _required_float("backtest", result.metrics, "max_drawdown"),
        "risk_gate_rejects": _required_int("backtest", result.metrics, "risk_gate_rejects"),
        "listing_exits": _exit_count(raw, weights),
        "source": "SYNTHETIC",
    }
    return stage, weights, result


def _read_master(cfg: AppConfig) -> pl.DataFrame:
    path = Path(cfg.data.root) / "bronze" / "security_master.parquet"
    if not path.is_file():
        raise E2ESmokeError("ingest: security master is missing")
    master = pl.read_parquet(path)
    if master.is_empty():
        raise E2ESmokeError("ingest: security master is empty")
    return master


def _exit_count(before: pl.DataFrame, after: pl.DataFrame) -> int:
    left = _gross_by_id(before)
    right = _gross_by_id(after)
    return sum(
        1 for sid in set(left) | set(right) if abs(left.get(sid, 0.0) - right.get(sid, 0.0)) > 1e-8
    )


def _gross_by_id(frame: pl.DataFrame) -> dict[str, float]:
    grouped = frame.group_by("security_id").agg(pl.col("target_weight").abs().sum().alias("gross"))
    return {str(row["security_id"]): float(row["gross"]) for row in grouped.iter_rows(named=True)}


def _portfolio(weights: pl.DataFrame, bars: pl.DataFrame, exits: int) -> dict[str, Any]:
    _when, book = _active_book(weights)
    ids, values = _aligned_weights(book)
    prices = _price_matrix(bars, ids, _when)
    try:
        _mcr, _cr, sig_p = component_risk(values, _covariance(prices))
    except ValueError as exc:
        raise E2ESmokeError(f"portfolio: {exc}") from exc
    return {
        "gross": _finite("portfolio", "gross", float(np.sum(np.abs(values)))),
        "net": _finite("portfolio", "net", float(np.sum(values))),
        "names": ids,
        "component_vol": _finite("portfolio", "component_vol", float(sig_p)),
        "listing_exits": exits,
    }


def _active_book(weights: pl.DataFrame) -> tuple[datetime, pl.DataFrame]:
    grouped = weights.group_by("event_time").agg(pl.col("target_weight").abs().sum().alias("gross"))
    active = grouped.filter(pl.col("gross") > 1e-8).sort("event_time")
    if active.is_empty():
        raise E2ESmokeError("portfolio: empty book")
    when = active.get_column("event_time").tail(1).item()
    if not isinstance(when, datetime):
        raise E2ESmokeError("portfolio: book time is not a datetime")
    book = weights.filter((pl.col("event_time") == when) & (pl.col("target_weight").abs() > 1e-12))
    if book.is_empty():
        raise E2ESmokeError("portfolio: empty book")
    return when, book


def _aligned_weights(book: pl.DataFrame) -> tuple[list[str], np.ndarray]:
    ordered = book.sort("security_id")
    ids = [str(value) for value in ordered.get_column("security_id").to_list()]
    values = ordered.get_column("target_weight").to_numpy().astype(float)
    if not np.all(np.isfinite(values)):
        raise E2ESmokeError("portfolio: non-finite weights")
    return ids, values


def _price_matrix(bars: pl.DataFrame, ids: list[str], when: datetime) -> np.ndarray:
    hist = bars.filter(pl.col("event_time") <= when)
    missing = [sid for sid in ids if hist.filter(pl.col("security_id") == sid).is_empty()]
    if missing:
        raise E2ESmokeError(f"portfolio: no prices for {missing[:8]}")
    wide = (
        hist.select(["event_time", "security_id", "close"])
        .pivot(on="security_id", index="event_time", values="close")
        .sort("event_time")
        .select(["event_time", *ids])
        .drop_nulls()
    )
    if wide.height < 3:
        raise E2ESmokeError("portfolio: fewer than 3 price rows")
    prices = wide.select(ids).to_numpy().astype(float)
    if prices.size == 0 or not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
        raise E2ESmokeError("portfolio: non-finite prices")
    return prices


def _covariance(prices: np.ndarray) -> np.ndarray:
    returns = prices[1:] / prices[:-1] - 1.0
    returns = returns[np.isfinite(returns).all(axis=1)]
    if returns.shape[0] < 2:
        raise E2ESmokeError("portfolio: fewer than 2 return rows")
    cov = np.atleast_2d(np.cov(returns, rowvar=False))
    cov = cov + (1e-12 * np.eye(cov.shape[0]))
    if not np.all(np.isfinite(cov)):
        raise E2ESmokeError("portfolio: non-finite covariance")
    return cov


def _risk(
    cfg: AppConfig, result: BacktestResult, weights: pl.DataFrame, bars: pl.DataFrame
) -> dict[str, Any]:
    returns = _simple_returns(result.equity)
    gated = apply_gate_stack(returns, _smoke_gates())
    if gated.scale.shape != returns.shape or not np.all(np.isfinite(gated.scale)):
        raise E2ESmokeError("risk: non-finite scale")
    var, es = _var_es(returns)
    return {
        "var": var,
        "es": es,
        "mean_scale": _finite("risk", "mean_scale", float(np.mean(gated.scale))),
        "last_scale": _finite("risk", "last_scale", float(gated.scale[-1])),
        "n_halt": int(gated.n_halt),
        "n_scaled": int(gated.n_scaled),
        "pretrade": _pretrade(cfg, weights, bars),
    }


def _smoke_gates() -> GateSpec:
    return GateSpec(
        vol_target=0.10,
        vol_lookback=8,
        vol_cap=4.0,
        dd_limit=0.25,
        es_limit=0.05,
        es_lookback=8,
        kelly_fraction=0.25,
        kelly_lookback=8,
        crc_alpha=0.05,
        crc_lookback=8,
        crash_lookback=5,
        crash_return=-0.2,
        stepm_enable=False,
    )


def _simple_returns(equity: pl.DataFrame) -> np.ndarray:
    require_populated("risk", equity, ("nav",))
    nav = equity.get_column("nav").to_numpy().astype(float)
    if nav.size < 2 or not np.all(np.isfinite(nav)) or np.any(nav <= 0.0):
        raise E2ESmokeError("risk: nav is empty or non-finite")
    returns = np.diff(nav) / nav[:-1]
    if returns.size == 0 or not np.all(np.isfinite(returns)):
        raise E2ESmokeError("risk: non-finite returns")
    return cast(np.ndarray, np.asarray(returns, dtype=np.float64))


def _var_es(returns: np.ndarray) -> tuple[float, float]:
    scored = portfolio_var_es(returns.reshape(-1, 1), np.array([1.0]))
    return (
        _finite("risk", "var", float(scored["var"])),
        _finite("risk", "es", float(scored["es"])),
    )


def _pretrade(cfg: AppConfig, weights: pl.DataFrame, bars: pl.DataFrame) -> str:
    when, book = _active_book(weights)
    ordered = book.with_columns(pl.col("target_weight").abs().alias("_abs")).sort(
        ["_abs", "security_id"], descending=[True, False]
    )
    row = ordered.row(0, named=True)
    weight = _finite("risk", "weight", float(row["target_weight"]))
    bar = _decision_bar(bars, when, str(row["security_id"]))
    price = _finite("risk", "price", float(bar["close"]))
    volume = _finite("risk", "volume", float(bar["volume"]))
    if price <= 0.0 or volume <= 0.0:
        raise E2ESmokeError("risk: price and volume must be positive")
    notional = abs(weight) * _PRETRADE_NAV
    participation = notional / (price * volume)
    side = OrderSide.BUY if weight > 0.0 else OrderSide.SELL
    order = Order(
        order_id="e2e-smoke-pretrade",
        security_id=str(row["security_id"]),
        symbol=str(row["security_id"]),
        side=side,
        quantity=abs(weight) * _PRETRADE_NAV / price,
        signal_time=when,
        decision_time=when,
        order_time=when,
    )
    try:
        check_order(
            order,
            nav=_PRETRADE_NAV,
            price=price,
            current_weight=0.0,
            gross_after=abs(weight),
            net_after=weight,
            participation=participation,
            predicted_vol=0.02,
            config=cfg,
            price_age_bars=0,
            model_age_hours=0.0,
        )
    except RiskGateRejected as exc:
        raise E2ESmokeError(f"risk: {exc}") from exc
    return "accepted"


def _decision_bar(bars: pl.DataFrame, when: datetime, security_id: str) -> dict[str, Any]:
    matched = bars.filter((pl.col("security_id") == security_id) & (pl.col("event_time") == when))
    if matched.height != 1:
        raise E2ESmokeError("risk: pretrade bar is missing")
    row = matched.row(0, named=True)
    if "close" not in row or "volume" not in row:
        raise E2ESmokeError("risk: pretrade bar is missing close or volume")
    return row


def _report(root: Path, stages: Mapping[str, Any], result: BacktestResult) -> dict[str, Any]:
    report_path = root / "metadata" / "reports" / "e2e_smoke.md"
    tear_path = root / "metadata" / "reports" / "e2e_tearsheet.md"
    write_report(report_path, "E2E smoke", dict(stages), synthetic=True)
    sheet = build_tearsheet(
        result.equity,
        fills=result.fills,
        label="E2E_SMOKE",
        synthetic=True,
    )
    require_finite("report", sheet)
    write_tearsheet_md(tear_path, sheet)
    _require_synthetic_text(report_path)
    _require_synthetic_text(tear_path)
    return {
        "smoke_sha256": hash_file(report_path),
        "tearsheet_sha256": hash_file(tear_path),
        "smoke_bytes": report_path.stat().st_size,
        "tearsheet_bytes": tear_path.stat().st_size,
    }


def _require_synthetic_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "SYNTHETIC" not in text or not text.strip():
        raise E2ESmokeError(f"report: {path.name} missing SYNTHETIC label")


def _receipt(cfg: AppConfig, stages: dict[str, Any]) -> dict[str, Any]:
    if tuple(stages) != _STAGE_NAMES:
        raise E2ESmokeError("receipt: stages are incomplete")
    body: dict[str, Any] = {
        "schema": "e2e_smoke.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "panel": {
            "n_assets": int(cfg.data.synthetic_n_assets),
            "n_days": int(cfg.data.synthetic_n_days),
            "seed": int(cfg.data.synthetic_seed),
        },
        "stages": stages,
    }
    require_finite("receipt", body)
    return {**body, "digest": hash_bytes(canonical_json_bytes(body))}


def _reject_empty_stages(stages: Mapping[str, Any]) -> None:
    for name, stage in stages.items():
        if not isinstance(stage, dict) or not stage:
            raise E2ESmokeError(f"{name}: empty")


def _required_float(stage: str, payload: Mapping[str, Any], key: str) -> float:
    if key not in payload:
        raise E2ESmokeError(f"{stage}: {key} is missing")
    return _finite(stage, key, _finite_number(payload[key]))


def _required_int(stage: str, payload: Mapping[str, Any], key: str) -> int:
    if key not in payload:
        raise E2ESmokeError(f"{stage}: {key} is missing")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise E2ESmokeError(f"{stage}: {key} is not an int")
    return int(value)


def _finite(stage: str, key: str, value: float) -> float:
    if not math.isfinite(value):
        raise E2ESmokeError(f"{stage}: {key} is non-finite")
    return 0.0 if value == 0.0 else value


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise E2ESmokeError("receipt: expected a number")
    return float(value)


def _path_free(note: str) -> bool:
    return "/" not in note and "\\" not in note and "\n" not in note


def _float_column_bad(frame: pl.DataFrame, name: str, dtype: pl.DataType) -> bool:
    return dtype in (pl.Float32, pl.Float64) and _series_nonfinite(frame.get_column(name))


def _series_nonfinite(column: pl.Series) -> bool:
    return bool(column.is_nan().any() or column.is_infinite().any())


def _walk(path: str, value: object, bad: list[str]) -> None:
    if _is_scalar(value):
        _note_scalar(path, value, bad)
        return
    _walk_nested(path, value, bad)


def _is_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, datetime, date)):
        return True
    return isinstance(value, (int, float, np.generic))


def _note_scalar(path: str, value: object, bad: list[str]) -> None:
    if isinstance(value, np.generic):
        _walk(path, value.item(), bad)
        return
    if isinstance(value, float) and not math.isfinite(value):
        bad.append(path)


def _walk_nested(path: str, value: object, bad: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _walk(f"{path}.{key}", item, bad)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(f"{path}[{index}]", item, bad)
        return
    if isinstance(value, np.ndarray):
        _walk_array(path, value, bad)
        return
    if isinstance(value, pl.DataFrame):
        if any(_float_column_bad(value, name, dtype) for name, dtype in value.schema.items()):
            bad.append(path)
        return
    raise E2ESmokeError(f"{path}: unsupported type {type(value).__name__}")


def _walk_array(path: str, value: np.ndarray, bad: list[str]) -> None:
    if not np.issubdtype(value.dtype, np.number):
        raise E2ESmokeError(f"{path}: unsupported ndarray dtype {value.dtype}")
    if value.size and not bool(np.isfinite(value.astype(float)).all()):
        bad.append(path)


@contextmanager
def _thread_limits() -> Iterator[None]:
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        with nullcontext():
            yield
        return
    with threadpool_limits(limits=1):
        yield


@contextmanager
def _isolated_tracking(db: Path) -> Iterator[None]:
    import mlflow

    import quant_fund.pipeline.train.ranking as ranking

    original = ranking.configure_tracking
    previous = mlflow.get_tracking_uri()
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    def _track(_uri: str | None = None) -> None:
        original(f"sqlite:///{db.resolve()}")

    ranking.configure_tracking = _track  # type: ignore[assignment]
    try:
        yield
    finally:
        ranking.configure_tracking = original
        mlflow.set_tracking_uri(previous)
