"""Adversarial probes for pmg_ardl."""

import numpy as np
import pytest

from quant_fund.models import pmg_ardl as pa


def test_pmg_rejects_dead_prior_arg():
    d = pa.synth_pmg(seed=0)
    with pytest.raises(ValueError, match="phi_true_prior"):
        pa.pmg_ardl(d["y_groups"], d["x_groups"], phi_true_prior=-0.3)


def test_mean_group_finiteness_and_k():
    yy = np.random.default_rng(0).normal(size=(3, 20))
    xx = np.random.default_rng(1).normal(size=(3, 20, 1))
    yy[0, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pa.mean_group(yy, xx)
    with pytest.raises(ValueError, match="panel too small"):
        pa.mean_group(np.zeros((3, 20)), np.zeros((3, 20, 0)))


def test_pmg_recovers_planted_theta():
    d = pa.synth_pmg(seed=4, n=8, t=160)
    out = pa.pmg_ardl(d["y_groups"], d["x_groups"])
    th = np.asarray(out["theta"])
    rel = np.linalg.norm(th - np.asarray(d["theta_true"])) / np.linalg.norm(d["theta_true"])
    assert rel < 0.2
    assert out["phi_mean"] < 0.0


def test_ssr_pooled_is_raw_sum():
    d = pa.synth_pmg(seed=2, n=5, t=80)
    out = pa.pmg_ardl(d["y_groups"], d["x_groups"])
    # recompute per-group raw SSR at the fitted theta and compare
    theta = np.asarray(out["theta"])
    total = 0.0
    for i in range(5):
        s, _, _ = pa._group_ssr(np.asarray(d["y_groups"])[i], np.asarray(d["x_groups"])[i], theta)
        total += s
    assert out["ssr_pooled"] == pytest.approx(total, rel=1e-8)
    assert out["ssr_dof_normalized"] > 0.0


def test_pmg_tighter_than_mg_on_shared_theta():
    d = pa.synth_pmg(seed=9, n=10, t=200)
    th_t = np.asarray(d["theta_true"])
    e_p = np.linalg.norm(np.asarray(pa.pmg_ardl(d["y_groups"], d["x_groups"])["theta"]) - th_t)
    e_m = np.linalg.norm(np.asarray(pa.mean_group(d["y_groups"], d["x_groups"])["theta_mg"]) - th_t)
    assert e_p <= e_m + 1e-9


def test_bench_smoke():
    out = pa.bench_pmg_ardl()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_err_pmg"] < 0.15


def test_mean_group_excludes_unidentified_theta():
    """A group with phi_i ~ 0 has no identified long-run theta; the old
    code divided by -1e-8 and averaged a ~1e8 garbage value in."""
    d = pa.synth_pmg(seed=20261231 + 295, n=8, t=160, k=1)
    yy = np.asarray(d["y_groups"]).copy()
    xx = np.asarray(d["x_groups"])
    # constant y -> dep = 0 -> every coefficient incl. phi_i is exactly
    # 0, so this group's theta_i is unidentified and must be excluded.
    yy[0] = 3.0
    out = pa.mean_group(yy, xx)
    ti = np.asarray(out["theta_i"])
    assert not np.isfinite(ti[0]).all()
    assert np.isfinite(out["theta_mg"]).all()
    # the surviving average must sit near the planted theta, not ~1e8
    assert np.abs(out["theta_mg"]).max() < 100.0


def test_mean_group_raises_when_no_group_identified():
    rng = np.random.default_rng(5)
    n, t, k = 4, 60, 1
    yy = np.full((n, t), 2.0)  # every group constant -> phi_i = 0
    xx = np.cumsum(rng.normal(0, 1, (n, t, k)), axis=1)
    with pytest.raises(ValueError, match="no long-run theta identified"):
        pa.mean_group(yy, xx)
