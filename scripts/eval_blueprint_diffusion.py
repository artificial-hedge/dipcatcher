"""One fixed, retrospective unconditional joint-path DDPM experiment.

This reads the existing graph evaluator's qualified Yahoo snapshot, with
the same selection/split eligibility. It downloads nothing. Generated
paths are always SYNTHETIC, including when trained on empirical inputs.
The previously inspected 2024--2025 tail is not fresh OOS evidence. No
parameter search, acceptance threshold, conditional forecast, TSTR,
extreme-tail, trading or SOTA claim is made.

The exclusive run directory is reserved before training. Source/code
drift aborts the receipt; reruns never replace an existing attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from threadpoolctl import threadpool_info, threadpool_limits

# Direct script invocation starts with scripts/ on sys.path; installed
# quant_fund and the checkout's scripts namespace must both be reachable.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import eval_blueprint_graph as graph  # noqa: E402

from quant_fund.metrics.energy_score import energy_score  # noqa: E402
from quant_fund.models.market_diffusion import (  # noqa: E402
    DiffusionConfig,
    MarketDiffusion,
    scenario_diagnostics,
)

MODEL_NAMES = ("ddpm", "multivariate_gaussian", "iid_marginal_empirical", "block_bootstrap")
SCENARIO_SEEDS = dict(zip(MODEL_NAMES, (7, 8, 9, 10), strict=True))


@dataclass(frozen=True)
class EvaluationConfig:
    """Predeclared immutable settings; no CLI tuning switches."""

    asset_count: int = 8
    horizon: int = 5
    ensemble_size: int = 128
    n_steps: int = 16
    hidden_dim: int = 32
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    seed: int = 7
    clip_denoised: bool = True
    gaussian_covariance_ddof: int = 1
    gaussian_eigenvalue_floor: float = 0.0


@dataclass(frozen=True)
class WindowPanel:
    paths: np.ndarray
    event_times: tuple[tuple[datetime, ...], ...]
    available_times: tuple[tuple[datetime, ...], ...]
    origin_times: tuple[tuple[datetime, ...], ...]
    audit: dict[str, Any]


def _clock(value: Any) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("source clocks must be timezone-aware datetimes")
    return value.astimezone(UTC)


def make_windows(
    phase: graph.PhasePanel,
    endpoint_times: np.ndarray,
    market_calendar: tuple[datetime, ...],
    *,
    asset_count: int = 8,
    horizon: int = 5,
) -> WindowPanel:
    """Nonoverlapping consecutive-session paths without bridging panel gaps.

    Targets come verbatim from prepare_panel. Return event clocks are the
    original label endpoints; availability is the latest source release
    across selected assets. Neither is replaced by the decision clock.
    """
    if isinstance(horizon, bool) or not 2 <= horizon <= 256:
        raise ValueError("valid horizon required")
    targets = np.asarray(phase.targets, dtype=float)
    if targets.ndim != 2 or not 2 <= asset_count <= targets.shape[1]:
        raise ValueError("valid target shape and asset count required")
    n = len(targets)
    if n != len(phase.timestamps) or endpoint_times.shape != (n, asset_count):
        raise ValueError("endpoint/target/origin axes must agree")
    if not np.isfinite(targets[:, :asset_count]).all():
        raise ValueError("finite return targets required")
    available = np.asarray(phase.target_available_times, dtype=object)
    feature_available = np.asarray(phase.feature_available_times, dtype=object)
    if available.shape != targets.shape or feature_available.shape != targets.shape:
        raise ValueError("source availability axes must agree")
    origins = tuple(_clock(t) for t in phase.timestamps)
    calendar = tuple(_clock(t) for t in market_calendar)
    if any(a >= b for a, b in zip(calendar[:-1], calendar[1:], strict=True)):
        raise ValueError("market calendar must be strictly ascending and unique")
    if any(a >= b for a, b in zip(origins[:-1], origins[1:], strict=True)):
        raise ValueError("decision clocks must be strictly ascending and unique")
    next_market = dict(zip(calendar[:-1], calendar[1:], strict=True))
    events: list[datetime] = []
    ready: list[datetime] = []
    for i, origin in enumerate(origins):
        row = [_clock(t) for t in endpoint_times[i]]
        if len(set(row)) != 1 or row[0] != next_market.get(origin):
            raise ValueError("each return must end at the next original market session")
        releases = [_clock(t) for t in available[i, :asset_count]]
        if any(t < row[0] for t in releases):
            raise ValueError("target cannot be available before its event endpoint")
        if any(_clock(t) > origin for t in feature_available[i, :asset_count]):
            raise ValueError("graph panel contains future feature availability")
        events.append(row[0])
        ready.append(max(releases))
    # Missing complete-case dates split the sequence. Discard remainders
    # separately within each segment instead of joining across a gap.
    boundaries = [0] + [i for i in range(1, n) if events[i - 1] != origins[i]] + [n]
    starts = [
        i
        for first, last in zip(boundaries[:-1], boundaries[1:], strict=True)
        for i in range(first, last - horizon + 1, horizon)
    ]
    if len(starts) < 2:
        raise ValueError("at least two complete nonoverlapping paths required")
    paths = np.stack([targets[i : i + horizon, :asset_count] for i in starts])

    def clocks(values: list[datetime] | tuple[datetime, ...]) -> tuple[tuple[datetime, ...], ...]:
        return tuple(tuple(values[i : i + horizon]) for i in starts)

    return WindowPanel(
        paths,
        clocks(events),
        clocks(ready),
        clocks(origins),
        {
            "source_decision_rows": n,
            "consecutive_segments": len(boundaries) - 1,
            "n_paths": len(paths),
            "used_return_rows": len(paths) * horizon,
            "discarded_return_rows": n - len(paths) * horizon,
            "first_return_event": events[starts[0]].isoformat(),
            "last_return_event": events[starts[-1] + horizon - 1].isoformat(),
            "event_basis": "original label_end_time_1, checked against source market calendar",
            "availability_basis": "maximum original endpoint-bar availability across selected assets",
        },
    )


def prepare_windows(
    data_root: Path, panel: graph.PreparedPanel, config: EvaluationConfig
) -> dict[str, WindowPanel]:
    """Add original clocks to the existing panel; do not rebuild its returns."""
    selected = panel.asset_ids[: config.asset_count]
    if len(selected) != config.asset_count:
        raise ValueError("graph selection does not contain the required assets")
    calendar = tuple(
        pl.read_parquet(data_root / "silver/bars.parquet", columns=["event_time"])
        .unique()
        .sort("event_time")["event_time"]
        .to_list()
    )
    labels = pl.read_parquet(
        data_root / "gold/labels.parquet",
        columns=["security_id", "event_time", "label_end_time_1"],
    ).filter(pl.col("security_id").is_in(selected))
    endpoints = {(a, t): end for a, t, end in labels.iter_rows()}
    result: dict[str, WindowPanel] = {}
    for name, phase in panel.phases.items():
        grid = np.asarray(
            [[endpoints.get((a, t)) for a in selected] for t in phase.timestamps], dtype=object
        )
        result[name] = make_windows(
            phase, grid, calendar, asset_count=config.asset_count, horizon=config.horizon
        )
    return result


def baseline_ensembles(
    training: np.ndarray, *, n: int = 128
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """All estimates use training paths only; all draws are SYNTHETIC."""
    paths = np.asarray(training, dtype=float)
    if paths.ndim != 3 or len(paths) < 8 or n < 2 or not np.isfinite(paths).all():
        raise ValueError("finite training paths and at least two ensemble draws required")
    flat = paths.reshape(len(paths), -1)
    mean, covariance = flat.mean(axis=0), np.cov(flat, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    gaussian = (
        np.random.default_rng(SCENARIO_SEEDS["multivariate_gaussian"])
        .normal(size=(n, flat.shape[1]))
        .dot(factor.T)
        + mean
    ).reshape(n, *paths.shape[1:])
    marginals = paths.reshape(-1, paths.shape[2])
    indices = np.random.default_rng(SCENARIO_SEEDS["iid_marginal_empirical"]).integers(
        0, len(marginals), size=(n, *paths.shape[1:])
    )
    iid = marginals[indices, np.arange(paths.shape[2])]
    block_indices = np.random.default_rng(SCENARIO_SEEDS["block_bootstrap"]).integers(
        0, len(paths), size=n
    )
    return (
        {
            "multivariate_gaussian": gaussian,
            "iid_marginal_empirical": iid,
            "block_bootstrap": paths[block_indices].copy(),
        },
        {"gaussian_mean": mean, "gaussian_covariance": covariance, "gaussian_factor": factor},
    )


def score_ensembles(
    ensembles: dict[str, np.ndarray], reference: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Same frozen ensemble and Euclidean return coordinates for every path."""
    real = np.asarray(reference, dtype=float)
    if real.ndim != 3 or len(real) < 2 or not np.isfinite(real).all():
        raise ValueError("finite held-out paths required")
    if set(ensembles) != set(MODEL_NAMES):
        raise ValueError("the four predeclared ensembles are required")
    sizes = {len(v) for v in ensembles.values()}
    if len(sizes) != 1 or next(iter(sizes)) < 2:
        raise ValueError("all models must have matched ensemble sizes >=2")
    scores: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    for name in MODEL_NAMES:
        draws = np.asarray(ensembles[name], dtype=float)
        if draws.ndim != 3 or draws.shape[1:] != real.shape[1:]:
            raise ValueError("all ensembles must have the held-out horizon/asset axes")
        flat = draws.reshape(len(draws), -1)
        losses = np.asarray([energy_score(flat, row.ravel()) for row in real])
        if not np.isfinite(losses).all():
            raise FloatingPointError("non-finite energy scores")
        scores[name] = losses
        reports[name] = {
            "energy_score": float(losses.mean()),
            "diagnostics": scenario_diagnostics(draws, real),
        }
    return (
        {
            "models": reports,
            "mean_ddpm_minus_baseline": {
                name: float(np.mean(scores["ddpm"] - scores[name])) for name in MODEL_NAMES[1:]
            },
            "n_reference_paths": len(real),
            "energy_score_definition": "unbiased off-diagonal U-statistic; Euclidean norm on flattened raw log returns",
            "lower_score_is_better": True,
            "acceptance_threshold": None,
        },
        scores,
    )


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, allow_pickle=False, **cast(dict[str, Any], arrays))
    return stream.getvalue()


def reserve_run(output: Path, protocol: dict[str, Any], code: dict[str, bytes]) -> None:
    """Reserve one attempt exclusively, including a failed/interrupted one."""
    output.mkdir(parents=True, exist_ok=False)
    files = {"protocol.json": (json.dumps(protocol, sort_keys=True, indent=2) + "\n").encode()}
    files.update({f"source_{name}.py": value for name, value in code.items()})
    for name, value in files.items():
        with (output / name).open("xb") as stream:
            stream.write(value)


def complete_run(output: Path, payload: dict[str, Any], artifacts: dict[str, bytes]) -> Path:
    """Never overwrite; bind protocol, code snapshots and all artifact bytes."""
    if not (output / "protocol.json").is_file():
        raise ValueError("predeclared reservation is required")
    if (output / "receipt.json").exists():
        raise FileExistsError("completed receipt already exists")
    protocol = json.loads((output / "protocol.json").read_text())
    if graph.canonical_hash(protocol) != payload.get("protocol_sha256"):
        raise ValueError("predeclared protocol changed before completion")
    for name, item in payload.get("code_files", {}).items():
        if graph.hash_file(output / f"source_{name}.py") != item["sha256"]:
            raise ValueError(f"code snapshot changed before completion: {name}")
    for name, value in artifacts.items():
        if Path(name).name != name or name in {"protocol.json", "receipt.json"}:
            raise ValueError("artifact names must be simple, unreserved filenames")
        with (output / name).open("xb") as stream:
            stream.write(value)
    hashes = {p.name: graph.hash_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    body = {**payload, "artifacts": hashes}
    receipt = output / "receipt.json"
    with receipt.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {**body, "receipt_sha256": graph.canonical_hash(body)},
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    return receipt


def verify_run(output: Path, *, check_current_sources: bool = True) -> dict[str, Any]:
    """Verify receipt/artifacts; current code/data drift fails by default."""
    payload: dict[str, Any] = json.loads((output / "receipt.json").read_text())
    digest = payload.pop("receipt_sha256")
    if graph.canonical_hash(payload) != digest:
        raise ValueError("receipt hash mismatch")
    if graph.canonical_hash(json.loads((output / "protocol.json").read_text())) != payload.get(
        "protocol_sha256"
    ):
        raise ValueError("predeclared protocol hash mismatch")
    artifacts = payload["artifacts"]
    actual_names = {p.name for p in output.iterdir() if p.is_file()} - {"receipt.json"}
    if actual_names != set(artifacts):
        raise ValueError("receipt artifact set drift")
    for name, expected in artifacts.items():
        if Path(name).name != name or graph.hash_file(output / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    for name, item in payload.get("code_files", {}).items():
        if graph.hash_file(output / f"source_{name}.py") != item["sha256"]:
            raise ValueError(f"code snapshot mismatch: {name}")
    if check_current_sources:
        for group in ("source_files", "code_files"):
            for name, item in payload.get(group, {}).items():
                if graph.hash_file(Path(item["path"])) != item["sha256"]:
                    raise ValueError(f"current {group} drift: {name}")
    return {**payload, "receipt_sha256": digest}


def _runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
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
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "polars", "torch", "scipy", "scikit-learn", "threadpoolctl")
        },
    }


def run_evaluation(data_root: Path, output: Path) -> Path:
    """Run exactly the fixed protocol; no score-dependent adaptation."""
    config, graph_config = EvaluationConfig(), graph.EvaluationConfig()
    source_paths = {
        "bars": data_root / "silver/bars.parquet",
        "universe": data_root / "silver/universe.parquet",
        "labels": data_root / "gold/labels.parquet",
    }
    model_source = sys.modules[MarketDiffusion.__module__].__file__
    metric_source = sys.modules[energy_score.__module__].__file__
    if model_source is None or metric_source is None:
        raise ValueError("model and metric source snapshots required")
    code_paths = {
        "script": Path(__file__),
        "panel": Path(graph.__file__),
        "market_diffusion": Path(model_source),
        "energy_score": Path(metric_source),
    }
    source_files = {
        name: {"path": str(path.resolve()), "sha256": graph.hash_file(path)}
        for name, path in source_paths.items()
    }
    code = {name: path.read_bytes() for name, path in code_paths.items()}
    code_files = {
        name: {"path": str(code_paths[name].resolve()), "sha256": hashlib.sha256(value).hexdigest()}
        for name, value in code.items()
    }
    protocol = {
        "schema_version": "blueprint_diffusion_fixed_protocol_v1",
        "config": asdict(config),
        "graph_panel_config": asdict(graph_config),
        "scenario_seeds": SCENARIO_SEEDS,
        "source_files": source_files,
        "code_files": code_files,
        "selection": "first8 of graph panel's training-only median-ADV selection; no reselection",
        "freeze": "all four unconditional ensembles frozen before validation/test scoring",
        "acceptance_threshold": None,
    }
    reserve_run(output, protocol, code)
    panel = graph.prepare_panel(data_root, graph_config)
    windows = prepare_windows(data_root, panel, config)
    cutoff = datetime.combine(
        date.fromisoformat(graph_config.train_end), datetime.max.time(), tzinfo=UTC
    )
    train = windows["train"]
    for name, window in windows.items():
        if name == "train" and any(t > cutoff for row in window.available_times for t in row):
            raise ValueError("training path unavailable by cutoff")
        if name != "train" and any(t <= cutoff for row in window.event_times for t in row):
            raise ValueError("held-out path overlaps training cutoff")
    runtime_before = _runtime()
    model = MarketDiffusion(
        DiffusionConfig(
            n_steps=config.n_steps,
            hidden_width=config.hidden_dim,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            seed=config.seed,
            clip_denoised=config.clip_denoised,
        )
    )
    with threadpool_limits(limits=1):
        model.fit(
            train.paths,
            asset_ids=panel.asset_ids[: config.asset_count],
            timestamps=train.event_times,
            available_times=train.available_times,
            cutoff=cutoff,
            data_source="qualified_local_yahoo_YAHOO_VENDOR_ADJ_snapshot",
            synthetic=False,
        )
        batch = model.sample_scenarios(config.ensemble_size, seed=SCENARIO_SEEDS["ddpm"])
        ensembles, baseline_parameters = baseline_ensembles(train.paths, n=config.ensemble_size)
        ensembles["ddpm"] = batch.paths
        params, fit_info = model._fitted()
        parameters = {
            "ddpm_mean": params.mean,
            "ddpm_scale": params.scale,
            "ddpm_lower": params.lower,
            "ddpm_upper": params.upper,
            "ddpm_betas": params.betas,
            **baseline_parameters,
        }
        for i, (weight, bias) in enumerate(params.layers):
            parameters[f"ddpm_layer{i}_weight"] = weight
            parameters[f"ddpm_layer{i}_bias"] = bias
        reports: dict[str, Any] = {}
        arrays = {f"ensemble_{name}": value for name, value in ensembles.items()}
        for name, window in windows.items():
            arrays[f"{name}_paths"] = window.paths
            for clock_name in ("event_times", "available_times", "origin_times"):
                arrays[f"{name}_{clock_name}"] = np.asarray(
                    [[t.isoformat() for t in row] for row in getattr(window, clock_name)]
                )
            if name == "train":
                continue
            reports[name], losses = score_ensembles(ensembles, window.paths)
            arrays.update({f"{name}_{key}_energy_scores": value for key, value in losses.items()})
        runtime_active = _runtime()
    model._fitted()  # parameter/fit metadata drift cannot receive an old model hash
    for files in (source_files, code_files):
        for name, item in files.items():
            if graph.hash_file(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"source/code changed during evaluation: {name}")
    payload = {
        **protocol,
        "protocol_sha256": graph.canonical_hash(protocol),
        "schema_version": "blueprint_diffusion_empirical_exploratory_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "asset_ids": panel.asset_ids[: config.asset_count],
        "graph_selected_asset_ids": panel.asset_ids,
        "selection": panel.selection,
        "panel_audit": panel.audit,
        "window_audit": {name: window.audit for name, window in windows.items()},
        "fit_info": asdict(fit_info),
        "model_parameter_array_sha256": graph.array_hash(parameters),
        "model_sha256": {
            "ddpm": fit_info.model_sha256,
            "multivariate_gaussian": graph.array_hash(baseline_parameters),
            "iid_marginal_empirical": graph.array_hash(
                {"training_marginals": train.paths.reshape(-1, config.asset_count)}
            ),
            "block_bootstrap": graph.array_hash({"training_paths": train.paths}),
        },
        "ensemble_sha256": {
            name: graph.array_hash({"paths": value}) for name, value in ensembles.items()
        },
        "results": reports,
        "runtime_before": runtime_before,
        "runtime_active": runtime_active,
        "training_inputs_synthetic": False,
        "generated_label": "SYNTHETIC",
        "generated_synthetic": True,
        "research_only": True,
        "market_realism_established": False,
        "conditional_forecast": False,
        "tstr_evidence": False,
        "extreme_tail_evidence": False,
        "live_pnl_claim": False,
        "sota_claim": False,
        "promote": False,
        "holdout_status": "previously_inspected",
        "evidence_class": "exploratory_retrospective_unconditional_distribution_scores",
        "limitations": [
            "Yahoo vendor-adjusted snapshot is not first-published PIT vintage data.",
            "Surviving vendor securities remain subject to survivorship bias despite train-only asset selection.",
            "Corporate-action completeness and true total-return adjustments remain unverified.",
            "Historical availability is reconstructed vendor close convention, not independently observed.",
            "Validation/test tail was previously inspected; these are reused retrospective slices, not fresh OOS evidence.",
            "Exact graph split is train2016-2021, validation2022-01-10..2023, test2024-01-10..2025-09-30; 2025 is incomplete.",
            "Complete-case graph date eligibility and nonoverlapping window remainders exclude observations; no imputation.",
            "Generator is unconditional and frozen; scoring reuses the same finite ensemble for every reference path.",
            "DDPM train-range clipping limits unseen extremes;1%/99% gaps on128 generated paths do not establish tail fidelity.",
            "Energy-score point estimates and distribution gaps have no pass thresholds, uncertainty intervals or multiplicity adjustment.",
            "This one small fixed run is not a hyperparameter search, conditional forecast, TSTR or learned market-realism demonstration.",
            "No execution, fees, net returns, live broker, institutional readiness or SOTA acceptance evidence is established.",
            "Snapshots/arrays bind inference and scoring; deterministic full retraining replay has not been performed.",
        ],
    }
    receipt = complete_run(
        output,
        payload,
        {"parameters.npz": _npz_bytes(parameters), "paths_and_scores.npz": _npz_bytes(arrays)},
    )
    verify_run(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/file_us_wide"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/metadata/blueprint_diffusion/seed7_fixed_v1")
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_run(args.out)
    else:
        receipt = run_evaluation(args.data_root, args.out)
        result = verify_run(receipt.parent)
    print(
        json.dumps(
            {
                "receipt": str((args.out / "receipt.json").resolve()),
                "receipt_sha256": result["receipt_sha256"],
                "asset_ids": result["asset_ids"],
                "window_audit": result["window_audit"],
                "energy_scores": {
                    name: {
                        model: report["energy_score"] for model, report in value["models"].items()
                    }
                    for name, value in result["results"].items()
                },
                "mean_ddpm_minus_baseline": {
                    name: value["mean_ddpm_minus_baseline"]
                    for name, value in result["results"].items()
                },
                "generated_label": "SYNTHETIC",
                "holdout_status": "previously_inspected",
                "sota_claim": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
