"""One predeclared, exploratory historical asset-GCN evaluation.

Run from this checkout with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. This reads
the existing local Yahoo-adjusted snapshot; it performs no downloads. All
four models see the same assets, eligible dates, features and labels. The
test tail was previously inspected: results are retrospective research
diagnostics, never fresh holdout, trading, promotion, or SOTA evidence.

The selection, architecture, optimizer, splits, baselines and CI settings
are fixed below before scores are computed. The script emits one exclusive
receipt plus hash-bound numpy parameter/prediction arrays. Parameters are
saved without pickle; an independent receipt verifier is not implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_info

from quant_fund.models.asset_graph import AssetGraphForecaster, GraphForecast, build_graph

FEATURE_NAMES = ("ret1", "ret5", "ret20", "vol20")
MODEL_NAMES = ("gcn", "node_only", "pooled_gaussian", "ridge_gaussian")


@dataclass(frozen=True)
class EvaluationConfig:
    train_start: str = "2016-01-01"
    train_end: str = "2021-12-31"
    validation_start: str = "2022-01-10"
    validation_end: str = "2023-12-31"
    test_start: str = "2024-01-10"
    test_end: str = "2025-09-30"
    asset_count: int = 24
    min_selection_rows: int = 1450
    min_phase_dates: int = 60
    seed: int = 7
    hidden_dim: int = 16
    epochs: int = 100
    learning_rate: float = 0.01
    min_scale: float = 0.02
    correlation_threshold: float = 0.3
    graph_top_k: int = 3
    ridge_alpha: float = 1.0
    embargo_sessions: int = 1
    bootstrap_block: int = 20
    bootstrap_samples: int = 1000


@dataclass(frozen=True)
class PhasePanel:
    features: np.ndarray
    targets: np.ndarray
    timestamps: tuple[datetime, ...]
    feature_available_times: np.ndarray
    target_available_times: np.ndarray


@dataclass(frozen=True)
class PreparedPanel:
    asset_ids: tuple[str, ...]
    phases: dict[str, PhasePanel]
    selection: list[dict[str, Any]]
    audit: dict[str, Any]
    source_paths: dict[str, Path]


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(arrays: dict[str, np.ndarray]) -> str:
    """Bind array name, shape, dtype and canonical numeric bytes."""
    digest = hashlib.sha256()
    for name, arr in sorted(arrays.items()):
        values = np.ascontiguousarray(arr, dtype="<f8")
        digest.update(json.dumps([name, values.shape, "float64_le"]).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _date_filter(column: str, first: str, last: str) -> pl.Expr:
    return pl.col(column).dt.date().is_between(date.fromisoformat(first), date.fromisoformat(last))


def _assert_frame(frame: pl.DataFrame, name: str) -> None:
    if frame.is_empty():
        raise ValueError(f"{name} is empty")
    if frame.select("security_id", "event_time").is_duplicated().any():
        raise ValueError(f"{name} has duplicate asset/time identities")
    if frame["security_id"].null_count() or frame["event_time"].null_count():
        raise ValueError(f"{name} has null asset/time identities")
    for clock in ("event_time", "available_time"):
        dtype = frame.schema[clock]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise ValueError(f"{name} {clock} must be timezone-aware UTC")
        if frame[clock].null_count():
            raise ValueError(f"{name} {clock} must be nonnull")
    if frame.filter(pl.col("available_time") < pl.col("event_time")).height:
        raise ValueError(f"{name} available_time cannot precede event_time")


def select_assets(universe: pl.DataFrame, config: EvaluationConfig) -> list[dict[str, Any]]:
    """Liquidity selection uses only training-window PIT-visible memberships."""
    _assert_frame(universe, "universe")
    eligible = universe.filter(
        _date_filter("event_time", config.train_start, config.train_end)
        & (pl.col("available_time") <= pl.col("event_time"))
        & pl.col("adv").is_finite()
        & (pl.col("adv") > 0)
    )
    selected = (
        eligible.group_by("security_id")
        .agg(pl.len().alias("n_training_memberships"), pl.col("adv").median().alias("median_adv"))
        .filter(pl.col("n_training_memberships") >= config.min_selection_rows)
        .sort(["median_adv", "security_id"], descending=[True, False])
        .head(config.asset_count)
    )
    if selected.height != config.asset_count:
        raise ValueError("insufficient assets meet the predeclared training-only selection")
    return selected.to_dicts()


def build_causal_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Features require 21 observed prices and no unpublished input bars."""
    _assert_frame(bars, "bars")
    if bars.filter(
        ~pl.col("close_total_return").is_finite() | (pl.col("close_total_return") <= 0)
    ).height:
        raise ValueError("prices must be finite and positive")
    ordered = bars.sort(["security_id", "event_time"])
    price = pl.col("close_total_return")
    ordered = ordered.with_columns(
        ((price / price.shift(1).over("security_id")).log()).alias("_log_return"),
        (price.shift(-1).over("security_id")).alias("_endpoint_price"),
        pl.col("event_time").shift(-1).over("security_id").alias("_endpoint_time"),
        pl.col("event_time").shift(20).over("security_id").alias("_feature_start_time"),
        pl.col("available_time").shift(-1).over("security_id").alias("_target_available_time"),
        *[(price / price.shift(h).over("security_id") - 1).alias(f"ret{h}") for h in (1, 5, 20)],
    )
    ordered = ordered.with_columns(
        pl.col("_log_return")
        .rolling_std(20, min_samples=20, ddof=1)
        .over("security_id")
        .alias("vol20"),
        pl.col("available_time")
        .cast(pl.Int64)
        .rolling_max(21, min_samples=21)
        .over("security_id")
        .cast(bars.schema["available_time"])
        .alias("_feature_available_time"),
    )
    return ordered


def _phase_panel(
    frame: pl.DataFrame, assets: tuple[str, ...], config: EvaluationConfig
) -> PhasePanel:
    complete = (
        frame.group_by("event_time").agg(pl.len().alias("n")).filter(pl.col("n") == len(assets))
    )
    balanced = (
        frame.join(complete.select("event_time"), on="event_time", how="inner")
        .with_columns(
            pl.col("security_id")
            .replace_strict(dict(zip(assets, range(len(assets)), strict=True)))
            .alias("_asset_position")
        )
        .sort(["event_time", "_asset_position"])
    )
    times = tuple(balanced["event_time"].unique(maintain_order=True).to_list())
    if len(times) < config.min_phase_dates:
        raise ValueError("insufficient complete eligible dates in a predeclared phase")
    shape = (len(times), len(assets))
    return PhasePanel(
        balanced.select(FEATURE_NAMES).to_numpy().reshape(*shape, len(FEATURE_NAMES)),
        balanced["future_log_return_1"].to_numpy().reshape(shape),
        times,
        np.asarray(balanced["_feature_available_time"].to_list(), dtype=object).reshape(shape),
        np.asarray(balanced["_target_available_time"].to_list(), dtype=object).reshape(shape),
    )


def prepare_panel(data_root: Path, config: EvaluationConfig = EvaluationConfig()) -> PreparedPanel:
    paths = {
        "bars": data_root / "silver/bars.parquet",
        "universe": data_root / "silver/universe.parquet",
        "labels": data_root / "gold/labels.parquet",
    }
    bars = pl.read_parquet(
        paths["bars"],
        columns=[
            "security_id",
            "event_time",
            "available_time",
            "source",
            "revision_id",
            "close_total_return",
        ],
    )
    universe = pl.read_parquet(
        paths["universe"], columns=["security_id", "event_time", "available_time", "adv"]
    )
    labels = pl.read_parquet(
        paths["labels"],
        columns=["security_id", "event_time", "label_end_time_1", "future_log_return_1"],
    )
    if set(bars["source"].unique().to_list()) != {"yahoo"} or set(
        bars["revision_id"].unique().to_list()
    ) != {"YAHOO_VENDOR_ADJ"}:
        raise ValueError(
            "this empirical protocol requires the declared Yahoo vendor-adjusted snapshot"
        )
    if labels.select("security_id", "event_time").is_duplicated().any():
        raise ValueError("labels have duplicate asset/time identities")
    selection = select_assets(universe, config)
    assets = tuple(row["security_id"] for row in selection)
    selected_bars = bars.filter(pl.col("security_id").is_in(assets))
    features = build_causal_features(selected_bars)
    membership = universe.filter(pl.col("available_time") <= pl.col("event_time")).select(
        "security_id", "event_time"
    )
    joined = features.join(membership, on=["security_id", "event_time"], how="inner").join(
        labels, on=["security_id", "event_time"], how="inner"
    )
    finite_label = joined.filter(
        pl.col("future_log_return_1").is_finite() & pl.col("label_end_time_1").is_not_null()
    )
    expected_label = (pl.col("_endpoint_price") / pl.col("close_total_return")).log()
    if finite_label.filter(
        (pl.col("label_end_time_1") != pl.col("_endpoint_time"))
        | ((pl.col("future_log_return_1") - expected_label).abs() > 1e-12)
    ).height:
        raise ValueError("stored one-session labels disagree with endpoint bars")
    # Missing sessions must not turn a multi-session gap into a one-session target.
    calendar = bars.select("event_time").unique().sort("event_time")
    calendar = calendar.with_columns(
        pl.col("event_time").shift(-1).alias("_next_market_time"),
        pl.col("event_time").shift(20).alias("_market_feature_start_time"),
    )
    joined = joined.join(calendar, on="event_time", how="left")
    eligible = joined.filter(
        pl.all_horizontal([pl.col(name).is_finite() for name in FEATURE_NAMES])
        & pl.col("future_log_return_1").is_finite()
        & (pl.col("_feature_available_time") <= pl.col("event_time"))
        & (pl.col("_target_available_time") > pl.col("event_time"))
        & (pl.col("label_end_time_1") == pl.col("_next_market_time"))
        & (pl.col("_feature_start_time") == pl.col("_market_feature_start_time"))
    )
    phases: dict[str, PhasePanel] = {}
    audit: dict[str, Any] = {
        "selected_member_label_rows": joined.height,
        "eligible_rows": eligible.height,
        "phase_counts": {},
    }
    for name, first, last, next_start in (
        ("train", config.train_start, config.train_end, config.validation_start),
        ("validation", config.validation_start, config.validation_end, config.test_start),
        ("test", config.test_start, config.test_end, None),
    ):
        phase = eligible.filter(
            _date_filter("event_time", first, last)
            & _date_filter("_target_available_time", first, last)
        )
        if next_start is not None:
            earlier = calendar.filter(
                pl.col("event_time").dt.date() < date.fromisoformat(next_start)
            )["event_time"].to_list()
            if len(earlier) <= config.embargo_sessions:
                raise ValueError("calendar cannot establish the predeclared split embargo")
            phase = phase.filter(pl.col("label_end_time_1") < earlier[-config.embargo_sessions])
        panel = _phase_panel(phase, assets, config)
        phases[name] = panel
        expected_dates = calendar.filter(_date_filter("event_time", first, last)).height
        audit["phase_counts"][name] = {
            "eligible_node_rows": phase.height,
            "balanced_dates": len(panel.timestamps),
            "balanced_node_rows": panel.targets.size,
            "observed_market_dates": expected_dates,
            "date_coverage": len(panel.timestamps) / expected_dates,
            "first_decision": panel.timestamps[0].isoformat(),
            "last_decision": panel.timestamps[-1].isoformat(),
        }
    return PreparedPanel(assets, phases, selection, audit, paths)


def paired_block_ci(
    difference: np.ndarray, *, block: int, samples: int, seed: int
) -> dict[str, float]:
    """Circular moving-block bootstrap over paired per-date score differences."""
    data = np.asarray(difference, dtype=float)
    if data.ndim != 1 or len(data) < 2 or not np.isfinite(data).all() or block < 1 or samples < 20:
        raise ValueError("finite paired date differences and valid bootstrap settings required")
    rng = np.random.default_rng(seed)
    block = min(block, len(data))
    starts = rng.integers(0, len(data), size=(samples, math.ceil(len(data) / block)))
    indices = (starts[..., None] + np.arange(block)) % len(data)
    resampled = data[indices.reshape(samples, -1)[:, : len(data)]].mean(axis=1)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return {
        "mean_gcn_minus_baseline": float(data.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def score_phase(
    predictions: dict[str, GraphForecast], targets: np.ndarray, config: EvaluationConfig
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if set(predictions) != set(MODEL_NAMES):
        raise ValueError("all four predeclared models must use identical node/date rows")
    reference = predictions["gcn"]
    if any(
        prediction.asset_ids != reference.asset_ids or prediction.timestamps != reference.timestamps
        for prediction in predictions.values()
    ):
        raise ValueError("prediction asset and date order must agree across all models")
    losses: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {"models": {}, "paired_date_block_ci": {}}
    for name, prediction in predictions.items():
        crps = prediction.crps(targets).mean(axis=1)
        nll = prediction.log_score(targets).mean(axis=1)
        losses[f"{name}_crps"] = crps
        losses[f"{name}_log_score"] = nll
        report["models"][name] = {"crps": float(crps.mean()), "log_score": float(nll.mean())}
    for baseline in MODEL_NAMES[1:]:
        report["paired_date_block_ci"][baseline] = {
            metric: paired_block_ci(
                losses[f"gcn_{metric}"] - losses[f"{baseline}_{metric}"],
                block=config.bootstrap_block,
                samples=config.bootstrap_samples,
                seed=config.seed,
            )
            for metric in ("crps", "log_score")
        }
    return report, losses


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, allow_pickle=False, **cast(dict[str, Any], arrays))
    return stream.getvalue()


def write_immutable_run(output: Path, payload: dict[str, Any], artifacts: dict[str, bytes]) -> Path:
    """Exclusive files; the receipt binds exact bytes of every side artifact."""
    if output.exists():
        raise FileExistsError(f"run output already exists: {output}")
    output.mkdir(parents=True)
    payload = {
        **payload,
        "artifacts": {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
    }
    digest = canonical_hash(payload)
    for name, data in artifacts.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    receipt = output / "receipt.json"
    with receipt.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {**payload, "receipt_sha256": digest}, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )
    return receipt


def run_evaluation(
    data_root: Path, output: Path, config: EvaluationConfig = EvaluationConfig()
) -> Path:
    if output.exists():
        raise FileExistsError(f"run output already exists: {output}")
    source_paths = {
        "bars": data_root / "silver/bars.parquet",
        "universe": data_root / "silver/universe.parquet",
        "labels": data_root / "gold/labels.parquet",
    }
    module = sys.modules[AssetGraphForecaster.__module__]
    model_source = module.__file__
    if model_source is None:
        raise ValueError("asset graph source path is missing")
    code_files = {"script": Path(__file__), "asset_graph": Path(model_source)}
    source_hashes = {name: hash_file(path) for name, path in source_paths.items()}
    code_hashes = {name: hash_file(path) for name, path in code_files.items()}
    panel = prepare_panel(data_root, config)
    train = panel.phases["train"]
    cutoff = datetime.combine(date.fromisoformat(config.train_end), datetime.max.time(), tzinfo=UTC)
    graph = build_graph(
        train.features[..., 0],
        asset_ids=panel.asset_ids,
        timestamps=train.timestamps,
        available_times=train.feature_available_times,
        as_of=cutoff,
        correlation_threshold=config.correlation_threshold,
        top_k=config.graph_top_k,
    )
    fit_kwargs: dict[str, Any] = dict(
        graph=graph,
        asset_ids=panel.asset_ids,
        timestamps=train.timestamps,
        feature_available_times=train.feature_available_times,
        target_available_times=train.target_available_times,
        feature_names=FEATURE_NAMES,
        fit_as_of=cutoff,
    )
    models = {
        name: AssetGraphForecaster(
            hidden_dim=config.hidden_dim,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            min_scale=config.min_scale,
            seed=config.seed,
            node_only=name == "node_only",
        ).fit(train.features, train.targets, **fit_kwargs)
        for name in ("gcn", "node_only")
    }
    x_mean = train.features.mean(axis=(0, 1))
    x_scale = np.maximum(train.features.std(axis=(0, 1)), 1e-12)
    x_train = ((train.features - x_mean) / x_scale).reshape(-1, len(FEATURE_NAMES))
    y_train = train.targets.ravel()
    ridge = Ridge(alpha=config.ridge_alpha).fit(x_train, y_train)
    pooled_mean = float(y_train.mean())
    scale_floor = config.min_scale * float(y_train.std())
    pooled_scale = max(float(y_train.std()), scale_floor, 1e-12)
    ridge_scale = max(
        float(np.sqrt(np.mean((y_train - ridge.predict(x_train)) ** 2))), scale_floor, 1e-12
    )
    arrays: dict[str, np.ndarray] = {
        "graph_adjacency": graph.adjacency,
        "asset_ids": np.asarray(panel.asset_ids),
        "feature_names": np.asarray(FEATURE_NAMES),
        "ridge_coef": ridge.coef_,
        "ridge_intercept": np.asarray([ridge.intercept_]),
        "ridge_scale": np.asarray([ridge_scale]),
        "ridge_x_mean": x_mean,
        "ridge_x_scale": x_scale,
        "pooled_parameters": np.asarray([pooled_mean, pooled_scale]),
    }
    model_hashes: dict[str, str] = {}
    for name, model in models.items():
        weights = model._weights
        if weights is None:
            raise ValueError("fitted graph weights are missing")
        parameters = {
            "w1": weights[0],
            "b1": weights[1],
            "w2": weights[2],
            "b2": weights[3],
            "x_mean": model._x_mean,
            "x_scale": model._x_scale,
            "y_mean_scale": np.asarray([model._y_mean, model._y_scale]),
        }
        arrays.update({f"{name}_{key}": value for key, value in parameters.items()})
        model_hashes[name] = canonical_hash(
            {
                "parameters": array_hash(parameters),
                "config": asdict(config),
                "node_only": model.node_only,
            }
        )
    model_hashes["ridge_gaussian"] = array_hash(
        {name: value for name, value in arrays.items() if name.startswith("ridge_")}
    )
    model_hashes["pooled_gaussian"] = array_hash({"parameters": arrays["pooled_parameters"]})
    prediction_arrays: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    feature_hashes: dict[str, str] = {}
    for phase_name, phase in panel.phases.items():
        feature_hashes[phase_name] = canonical_hash(
            {
                "arrays": array_hash({"features": phase.features, "targets": phase.targets}),
                "timestamps": [t.isoformat() for t in phase.timestamps],
                "asset_ids": panel.asset_ids,
                "feature_available_times": [
                    [t.isoformat() for t in row] for row in phase.feature_available_times
                ],
                "target_available_times": [
                    [t.isoformat() for t in row] for row in phase.target_available_times
                ],
            }
        )
        if phase_name == "train":
            continue
        prediction_kwargs: dict[str, Any] = dict(
            graph=graph,
            asset_ids=panel.asset_ids,
            timestamps=phase.timestamps,
            feature_available_times=phase.feature_available_times,
            feature_names=FEATURE_NAMES,
        )
        predictions = {
            name: model.predict_from_graph(phase.features, **prediction_kwargs)
            for name, model in models.items()
        }
        ridge_mean = ridge.predict(
            ((phase.features - x_mean) / x_scale).reshape(-1, len(FEATURE_NAMES))
        ).reshape(phase.targets.shape)
        predictions["ridge_gaussian"] = GraphForecast(
            ridge_mean, np.full_like(ridge_mean, ridge_scale), panel.asset_ids, phase.timestamps
        )
        predictions["pooled_gaussian"] = GraphForecast(
            np.full_like(ridge_mean, pooled_mean),
            np.full_like(ridge_mean, pooled_scale),
            panel.asset_ids,
            phase.timestamps,
        )
        reports[phase_name], losses = score_phase(predictions, phase.targets, config)
        prediction_arrays.update({f"{phase_name}_{key}": value for key, value in losses.items()})
        prediction_arrays[f"{phase_name}_targets"] = phase.targets
        prediction_arrays[f"{phase_name}_timestamps"] = np.asarray(
            [t.isoformat() for t in phase.timestamps]
        )
        for name, prediction in predictions.items():
            prediction_arrays[f"{phase_name}_{name}_mean"] = prediction.mean
            prediction_arrays[f"{phase_name}_{name}_scale"] = prediction.scale
    graph_hash = canonical_hash(
        {
            "adjacency": array_hash({"graph": graph.adjacency}),
            "asset_ids": graph.asset_ids,
            "as_of": graph.as_of.isoformat(),
            "observation_times": [t.isoformat() for t in graph.observation_times],
        }
    )
    if source_hashes != {name: hash_file(path) for name, path in source_paths.items()}:
        raise ValueError("source data changed during evaluation; refusing a receipt")
    if code_hashes != {name: hash_file(path) for name, path in code_files.items()}:
        raise ValueError("evaluation code changed during evaluation; refusing a receipt")
    source_files = {
        name: {"path": str(path.resolve()), "sha256": source_hashes[name]}
        for name, path in panel.source_paths.items()
    }
    payload = {
        "schema_version": "blueprint_graph_empirical_exploratory_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "source_files": source_files,
        "code_sha256": code_hashes,
        "feature_names": FEATURE_NAMES,
        "feature_sha256": feature_hashes,
        "graph_sha256": graph_hash,
        "model_sha256": model_hashes,
        "selection": panel.selection,
        "selection_basis": "training-only PIT-visible median ADV; minimum training memberships; no test-driven reselection",
        "audit": panel.audit,
        "results": reports,
        "training_loss_first_last": {
            name: [model.loss_history[0], model.loss_history[-1]] for name, model in models.items()
        },
        "source": "yahoo",
        "revision_id": "YAHOO_VENDOR_ADJ",
        "synthetic": False,
        "research_only": True,
        "live_pnl_claim": False,
        "promote": False,
        "sota_claim": False,
        "holdout_status": "previously_inspected",
        "evidence_class": "exploratory_retrospective_forecast_scores",
        "availability_basis": "reconstructed vendor close-publication convention",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "threadpools": [
                {key: pool.get(key) for key in ("internal_api", "num_threads", "prefix")}
                for pool in threadpool_info()
            ],
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "polars", "torch", "scipy", "scikit-learn")
            },
        },
        "limitations": [
            "Vendor-adjusted snapshot is not first-published PIT vintage data.",
            "Existing vendor pool contains surviving securities; training-only selection does not remove survivorship bias.",
            "Gold close_total_return inherits unverified corporate-action completeness; identity adjustments do not establish true total returns.",
            "Historical availability is reconstructed, not independently observed.",
            "Validation/test tail was already inspected by prior research; this is not fresh OOS evidence.",
            "Complete-case date coverage excludes missing or ineligible node rows; no imputation or future asset reselection.",
            "Graph is static; node-specific joint density and dynamic spatiotemporal modeling are absent.",
            "CI is a paired circular moving-block diagnostic with 20-session blocks; baseline/score comparisons are not multiplicity-adjusted.",
            "No execution, costs, net returns, live broker, or SOTA acceptance evidence is established.",
            "Receipt hashes and numpy arrays bind this run; independent receipt verification/retraining replay is still required.",
        ],
    }
    return write_immutable_run(
        output,
        payload,
        {"models.npz": _npz_bytes(arrays), "predictions.npz": _npz_bytes(prediction_arrays)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/file_us_wide"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_evaluation(args.data_root, args.out)
    result = json.loads(receipt.read_text())
    print(
        json.dumps(
            {
                "receipt": str(receipt.resolve()),
                "receipt_sha256": result["receipt_sha256"],
                "results": result["results"],
                "holdout_status": result["holdout_status"],
                "sota_claim": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
