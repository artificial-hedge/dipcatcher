"""Read and independently recompute saved graph forecast score evidence.

This verifies artifact integrity, parameter identities and Gaussian score
arithmetic. It does not retrain models, replay source-data feature construction,
or independently establish historical data rights/availability. The receipt's
legacy ``log_score`` field means Gaussian negative log density, lower is better;
the UI exposes it as ``gaussian_nll`` to distinguish the metrics module's
higher-is-better logarithmic score.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from scipy.special import ndtr

_MODELS = ("gcn", "node_only", "pooled_gaussian", "ridge_gaussian")
_MAX_ARTIFACT_BYTES = 32_000_000


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        values = np.ascontiguousarray(value, dtype="<f8")
        digest.update(json.dumps([name, values.shape, "float64_le"]).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _arrays(path: Path) -> dict[str, np.ndarray]:
    # Bound decompressed size before numpy materializes arrays; refuse pickle.
    with ZipFile(path) as archive:
        if sum(row.file_size for row in archive.infolist()) > _MAX_ARTIFACT_BYTES:
            raise ValueError("graph artifact exceeds the decompressed resource budget")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def read_graph_evidence(run: Path) -> dict[str, Any]:
    """Return fail-closed graph/score evidence suitable for a local research UI."""
    root = run.resolve()
    receipt = (root / "receipt.json").resolve()
    if receipt.parent != root:
        raise ValueError("graph receipt escapes its artifact root")
    if receipt.stat().st_size > 1_000_000:
        raise ValueError("graph receipt exceeds the resource budget")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("graph receipt must be an object")
    receipt_hash = payload.pop("receipt_sha256", None)
    if receipt_hash != _digest(payload):
        raise ValueError("graph receipt hash mismatch")
    if payload.get("schema_version") != "blueprint_graph_empirical_exploratory_v1":
        raise ValueError("unsupported graph receipt schema")
    for key, value in {
        "research_only": True,
        "live_pnl_claim": False,
        "promote": False,
        "sota_claim": False,
    }.items():
        if payload.get(key) is not value:
            raise ValueError("graph receipt honesty flags invalid")
    artifacts = payload["artifacts"]
    if set(artifacts) != {"models.npz", "predictions.npz"}:
        raise ValueError("unexpected graph artifacts")
    for name, expected in artifacts.items():
        path = (root / name).resolve()
        if path.parent != root or path.stat().st_size > _MAX_ARTIFACT_BYTES:
            raise ValueError("graph artifact path or resource budget invalid")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("graph artifact hash mismatch")
    parameters = _arrays(root / "models.npz")
    predictions = _arrays(root / "predictions.npz")
    asset_ids = parameters["asset_ids"].tolist()
    adjacency = parameters["graph_adjacency"]
    if (
        not 2 <= len(asset_ids) <= 128
        or len(set(asset_ids)) != len(asset_ids)
        or any(not isinstance(value, str) or not value for value in asset_ids)
        or adjacency.shape != (len(asset_ids), len(asset_ids))
        or not np.isfinite(adjacency).all()
        or np.any((adjacency < 0) | (adjacency > 1))
        or not np.allclose(adjacency, adjacency.T)
    ):
        raise ValueError("invalid graph asset/adjacency dimensions")
    for model in _MODELS:
        if model in {"gcn", "node_only"}:
            arrays = {
                key.removeprefix(f"{model}_"): value
                for key, value in parameters.items()
                if key.startswith(f"{model}_")
            }
            expected_keys = {"w1", "b1", "w2", "b2", "x_mean", "x_scale", "y_mean_scale"}
            if set(arrays) != expected_keys or any(
                not np.isfinite(value).all() for value in arrays.values()
            ):
                raise ValueError("invalid graph model parameter schema")
            hidden = payload["config"]["hidden_dim"]
            features = len(payload["feature_names"])
            if (
                arrays["w1"].shape != (hidden, features)
                or arrays["b1"].shape != (hidden,)
                or arrays["w2"].shape != (2, hidden)
                or arrays["b2"].shape != (2,)
                or arrays["x_mean"].shape != (features,)
                or arrays["x_scale"].shape != (features,)
                or np.any(arrays["x_scale"] <= 0)
                or arrays["y_mean_scale"].shape != (2,)
                or arrays["y_mean_scale"][1] <= 0
            ):
                raise ValueError("invalid graph model parameter dimensions")
            actual = _digest(
                {
                    "parameters": _array_digest(arrays),
                    "config": payload["config"],
                    "node_only": model == "node_only",
                }
            )
        elif model == "ridge_gaussian":
            actual = _array_digest(
                {key: value for key, value in parameters.items() if key.startswith("ridge_")}
            )
        else:
            actual = _array_digest({"parameters": parameters["pooled_parameters"]})
        if actual != payload["model_sha256"][model]:
            raise ValueError("graph model parameter identity mismatch")
    scores: dict[str, Any] = {}
    for phase in ("validation", "test"):
        target = predictions[f"{phase}_targets"]
        if target.ndim != 2 or target.shape[1] != len(asset_ids) or not np.isfinite(target).all():
            raise ValueError("invalid graph score target dimensions")
        if not 1 <= len(target) <= 10_000:
            raise ValueError("graph score row budget exceeded")
        scores[phase] = {}
        for model in _MODELS:
            mean = predictions[f"{phase}_{model}_mean"]
            scale = predictions[f"{phase}_{model}_scale"]
            if (
                mean.shape != target.shape
                or scale.shape != target.shape
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or np.any(scale <= 0)
            ):
                raise ValueError("invalid graph prediction dimensions")
            z = (target - mean) / scale
            nll = float(np.mean(0.5 * math.log(2 * math.pi) + np.log(scale) + 0.5 * z**2))
            phi = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
            crps = float(
                np.mean(scale * (z * (2 * ndtr(z) - 1) + 2 * phi - 1 / math.sqrt(math.pi)))
            )
            recorded = payload["results"][phase]["models"][model]
            if not math.isclose(nll, recorded["log_score"], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError("graph Gaussian NLL did not reproduce")
            if not math.isclose(crps, recorded["crps"], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError("graph CRPS did not reproduce")
            scores[phase][model] = {"crps": crps, "gaussian_nll": nll}
    return {
        "schema_version": "blueprint_graph_ui_v1",
        "receipt_sha256": receipt_hash,
        "source": payload["source"],
        "synthetic": payload["synthetic"],
        "evidence_class": payload["evidence_class"],
        "holdout_status": payload["holdout_status"],
        "asset_ids": asset_ids,
        "edges": [
            {"source": asset_ids[i], "target": asset_ids[j], "weight": float(adjacency[i, j])}
            for i in range(len(asset_ids))
            for j in range(i + 1, len(asset_ids))
            if adjacency[i, j] > 0
        ],
        "graph_basis": "frozen training-only absolute return correlation, symmetric top-k union",
        "scores": scores,
        "score_directions": {"crps": "lower_is_better", "gaussian_nll": "lower_is_better"},
        "audit": payload["audit"],
        "paired_ci_recorded": payload["results"]["test"]["paired_date_block_ci"],
        "limitations": payload["limitations"],
        "verification": {
            "receipt_and_artifact_hashes": True,
            "parameter_identities": True,
            "score_arithmetic_reproduced": True,
            "forecast_predictions_recomputed": False,
            "training_repeated": False,
            "historical_availability_independently_verified": False,
        },
        "research_only": True,
        "live_pnl_claim": False,
        "promote": False,
        "sota_claim": False,
    }
