"""Artifact checks and independent Gaussian arithmetic, using synthetic arrays."""

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.stats import norm

from quant_fund.api.app import app
from quant_fund.research.blueprint_graph_evidence import (
    _array_digest,
    _digest,
    read_graph_evidence,
)


def _write_receipt(root: Path, payload):
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    (root / "receipt.json").write_text(
        json.dumps({**unsigned, "receipt_sha256": _digest(unsigned)})
    )


def _fixture(root: Path):
    root.mkdir()
    parameters = {
        "asset_ids": np.array(["A", "B"]),
        "graph_adjacency": np.array([[0, 0.5], [0.5, 0]]),
        "ridge_coef": np.zeros(2),
        "ridge_intercept": np.zeros(1),
        "ridge_scale": np.ones(1),
        "ridge_x_mean": np.zeros(2),
        "ridge_x_scale": np.ones(2),
        "pooled_parameters": np.array([0, 1]),
    }
    model_hashes = {}
    config = {"hidden_dim": 2}
    for model in ("gcn", "node_only"):
        arrays = {
            "w1": np.zeros((2, 2)),
            "w2": np.zeros((2, 2)),
            "b1": np.zeros(2),
            "b2": np.zeros(2),
            "x_mean": np.zeros(2),
            "x_scale": np.ones(2),
            "y_mean_scale": np.array([0, 1]),
        }
        parameters.update({f"{model}_{key}": value for key, value in arrays.items()})
        model_hashes[model] = _digest(
            {
                "parameters": _array_digest(arrays),
                "config": config,
                "node_only": model == "node_only",
            }
        )
    model_hashes["ridge_gaussian"] = _array_digest(
        {key: value for key, value in parameters.items() if key.startswith("ridge_")}
    )
    model_hashes["pooled_gaussian"] = _array_digest({"parameters": parameters["pooled_parameters"]})
    predictions = {}
    results = {}
    # Unit target residual: use scipy's independent log-density and Gaussian
    # CRPS identity rather than the reader or GraphForecast scoring methods.
    nll = -float(norm.logpdf(1))
    crps = float(2 * norm.cdf(1) - 1 + 2 * norm.pdf(1) - 1 / math.sqrt(math.pi))
    for phase in ("validation", "test"):
        predictions[f"{phase}_targets"] = np.ones((3, 2))
        models = {}
        for model in model_hashes:
            predictions[f"{phase}_{model}_mean"] = np.zeros((3, 2))
            predictions[f"{phase}_{model}_scale"] = np.ones((3, 2))
            models[model] = {"crps": crps, "log_score": nll}
        results[phase] = {"models": models, "paired_date_block_ci": {}}
    artifacts = {}
    for name, arrays in (("models.npz", parameters), ("predictions.npz", predictions)):
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        (root / name).write_bytes(stream.getvalue())
        artifacts[name] = hashlib.sha256(stream.getvalue()).hexdigest()
    payload = {
        "schema_version": "blueprint_graph_empirical_exploratory_v1",
        "research_only": True,
        "live_pnl_claim": False,
        "promote": False,
        "sota_claim": False,
        "artifacts": artifacts,
        "model_sha256": model_hashes,
        "config": config,
        "feature_names": ["f1", "f2"],
        "results": results,
        "source": "SYNTHETIC_evidence_fixture",
        "synthetic": True,
        "evidence_class": "synthetic_correctness_only",
        "holdout_status": "synthetic_fixture",
        "audit": {},
        "limitations": ["Synthetic fixture is not a trained model or market dataset."],
    }
    _write_receipt(root, payload)
    return payload


def test_scores_reproduce_and_verification_limits_remain_explicit(tmp_path):
    run = tmp_path / "run"
    _fixture(run)
    result = read_graph_evidence(run)
    assert result["scores"]["test"]["gcn"]["gaussian_nll"] == pytest.approx(-norm.logpdf(1))
    assert result["verification"]["score_arithmetic_reproduced"]
    assert not result["verification"]["forecast_predictions_recomputed"]
    assert not result["verification"]["training_repeated"]
    assert result["edges"] == [{"source": "A", "target": "B", "weight": 0.5}]


@pytest.mark.parametrize("change", ["score", "model", "honesty"])
def test_self_rehashed_receipt_cannot_fake_computed_checks(tmp_path, change):
    run = tmp_path / "run"
    payload = _fixture(run)
    if change == "score":
        payload["results"]["test"]["models"]["gcn"]["crps"] = 0.0
    elif change == "model":
        payload["model_sha256"]["gcn"] = "0" * 64
    else:
        payload["sota_claim"] = True
    _write_receipt(run, payload)
    with pytest.raises(ValueError):
        read_graph_evidence(run)


def test_changed_artifact_and_symlink_escape_reject(tmp_path):
    run = tmp_path / "run"
    _fixture(run)
    (run / "models.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact hash"):
        read_graph_evidence(run)
    outside = tmp_path / "outside.json"
    (run / "receipt.json").rename(outside)
    (run / "receipt.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        read_graph_evidence(run)


def test_api_dashboard_and_graph_reader_use_bounded_verified_root(tmp_path, monkeypatch):
    import quant_fund.api.blueprint as api

    _fixture(tmp_path / "fixture")
    monkeypatch.setattr(api, "_GRAPH_RUNS", tmp_path)
    client = TestClient(app)
    response = client.get("/v1/blueprint/graph/fixture")
    assert response.status_code == 200, response.text
    assert response.json()["synthetic"]
    assert client.get("/v1/blueprint/graph/missing").status_code == 404
    assert client.get("/v1/blueprint/graph/illegal.run").status_code == 422
    dashboard = client.get("/v1/blueprint/dashboard")
    assert dashboard.status_code == 200
    assert "does not retrain" in dashboard.text
