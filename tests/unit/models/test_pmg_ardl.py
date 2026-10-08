"""Unit tests for quant_fund.models.pmg_ardl."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.pmg_ardl import (
    bench_pmg_ardl,
    mean_group,
    pmg_ardl,
    synth_pmg,
)


def test_pmg_theta_close_to_truth() -> None:
    d = synth_pmg(seed=10)
    out = pmg_ardl(np.asarray(d["y_groups"]), np.asarray(d["x_groups"]))
    th = np.asarray(out["theta"])
    th_t = np.asarray(d["theta_true"])
    assert float(np.linalg.norm(th - th_t) / np.linalg.norm(th_t)) < 0.15


def test_pmg_beats_or_matches_mg() -> None:
    d = synth_pmg(seed=11)
    yy = np.asarray(d["y_groups"])
    xx = np.asarray(d["x_groups"])
    th_t = np.asarray(d["theta_true"])
    e_p = float(np.linalg.norm(np.asarray(pmg_ardl(yy, xx)["theta"]) - th_t))
    e_m = float(np.linalg.norm(np.asarray(mean_group(yy, xx)["theta_mg"]) - th_t))
    assert e_p <= e_m + 0.05


def test_phi_mean_negative() -> None:
    d = synth_pmg(seed=12)
    out = pmg_ardl(np.asarray(d["y_groups"]), np.asarray(d["x_groups"]))
    assert out["phi_mean"] < 0.0
    assert out["phi_mean"] > -1.0


def test_mg_shape() -> None:
    d = synth_pmg(seed=13, n=5, k=2)
    out = mean_group(np.asarray(d["y_groups"]), np.asarray(d["x_groups"]))
    assert np.asarray(out["theta_mg"]).shape == (2,)
    assert np.asarray(out["theta_i"]).shape == (5, 2)


def test_deterministic() -> None:
    d = synth_pmg(seed=14)
    args = (np.asarray(d["y_groups"]), np.asarray(d["x_groups"]))
    a = pmg_ardl(*args)
    b = pmg_ardl(*args)
    assert np.allclose(np.asarray(a["theta"]), np.asarray(b["theta"]))
    assert a["phi_mean"] == b["phi_mean"]


def test_fail_closed() -> None:
    d = synth_pmg(seed=15)
    yy = np.asarray(d["y_groups"])
    xx = np.asarray(d["x_groups"])
    with pytest.raises(ValueError):
        pmg_ardl(yy, xx[:, :, :1].reshape(yy.shape[0], yy.shape[1], 1)[:, :5, :])
    with pytest.raises(ValueError):
        pmg_ardl(yy.reshape(-1), xx)
    with pytest.raises(ValueError):
        mean_group(np.full_like(yy, np.nan), xx)


def test_bench_score() -> None:
    out = bench_pmg_ardl()
    assert out["synthetic_score"] == 1.0
    assert set(out) == {
        "synthetic_theta_hat",
        "synthetic_theta_true",
        "synthetic_theta_mg",
        "synthetic_err_pmg",
        "synthetic_err_mg",
        "synthetic_phi_mean",
        "synthetic_score",
    }
