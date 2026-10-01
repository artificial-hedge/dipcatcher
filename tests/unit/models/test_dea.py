"""Unit tests for quant_fund.models.dea."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dea import bench_dea, dea_ccr, synth_dea


def test_inefficient_units_score_low() -> None:
    d = synth_dea(seed=1)
    out = dea_ccr(d["inputs"], d["outputs"])
    scores = np.asarray(out["_scores"])
    idx = np.asarray(d["idx"])
    assert np.all(scores[idx] < 0.8)
    assert out["n_efficient"] >= 10


def test_self_referent_efficiency() -> None:
    # A single frontier unit should always score 1
    xx = np.array([[1.0], [2.0], [3.0], [4.0]])
    yy = np.array([[1.0], [2.0], [3.0], [4.0]])
    out = dea_ccr(xx, yy)
    scores = np.asarray(out["_scores"])
    assert scores[3] == pytest.approx(1.0, abs=1e-6)


def test_schema() -> None:
    d = synth_dea(seed=2)
    out = dea_ccr(d["inputs"], d["outputs"])
    assert set(out) == {
        "efficiency_mean",
        "efficiency_min",
        "n_efficient",
        "n_referents_mean",
        "dispersion",
        "_scores",
    }


def test_determinism() -> None:
    d = synth_dea(seed=5)
    a = dea_ccr(d["inputs"], d["outputs"])
    b = dea_ccr(d["inputs"], d["outputs"])
    assert np.array_equal(np.asarray(a["_scores"]), np.asarray(b["_scores"]))
    assert {k: v for k, v in a.items() if k != "_scores"} == {
        k: v for k, v in b.items() if k != "_scores"
    }


def test_vrs_variant_runs() -> None:
    d = synth_dea(seed=3)
    out = dea_ccr(d["inputs"], d["outputs"], vrs=True)
    scores = np.asarray(out["_scores"])
    assert np.all(scores <= 1.0 + 1e-6)
    assert np.all(scores > 0.0)


def test_validation() -> None:
    with pytest.raises(ValueError):
        dea_ccr(np.ones((2, 1)), np.ones((2, 1)))  # too few DMUs
    with pytest.raises(ValueError):
        dea_ccr(np.zeros((10, 1)), np.ones((10, 1)))  # non-positive
    with pytest.raises(ValueError):
        dea_ccr(np.full((10, 1), np.nan), np.ones((10, 1)))
    with pytest.raises(ValueError):
        dea_ccr(np.ones((10, 1)), np.ones((9, 1)))  # row mismatch


def test_bench_keys_and_pass() -> None:
    out = bench_dea()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_efficiency_mean",
        "synthetic_efficiency_min",
        "synthetic_n_efficient",
        "synthetic_worst_match",
        "synthetic_dispersion",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
