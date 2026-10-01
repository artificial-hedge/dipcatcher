"""Tests for optimal transport (models/optimal_transport.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.optimal_transport import (
    bench_optimal_transport,
    ot_plan,
    sinkhorn,
    synth_measures,
    wasserstein_barycenter,
    wasserstein_distance,
)


@pytest.fixture
def measures():
    return synth_measures(n_bins=40, seed=3)


def _cost(s):
    d = s[:, None] - s[None, :]
    return d * d


def test_sinkhorn_marginals(measures):
    s = np.asarray(measures["support"])
    c = _cost(s)
    p = sinkhorn(np.asarray(measures["m0"]), np.asarray(measures["m1"]), c)
    plan = np.asarray(p["plan"])
    assert np.allclose(
        plan.sum(axis=1), np.asarray(measures["m0"]) / measures["m0"].sum(), atol=1e-4
    )
    assert np.allclose(
        plan.sum(axis=0), np.asarray(measures["m1"]) / measures["m1"].sum(), atol=1e-4
    )
    assert p["marginal_violation"] < 1e-5


def test_ot_monotone_in_translation(measures):
    s = np.asarray(measures["support"])
    c = _cost(s)
    m0 = np.asarray(measures["m0"])
    d0 = sinkhorn(m0, m0, c)["ot_cost"]
    d1 = sinkhorn(m0, np.asarray(measures["m1"]), c)["ot_cost"]
    d2 = sinkhorn(m0, np.asarray(measures["m2"]), c)["ot_cost"]
    assert d0 <= d1 <= d2


def test_wasserstein_cloud():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = rng.normal(2.0, 1, 200)
    near = wasserstein_distance(x, x + 0.1, eps=0.05)
    far = wasserstein_distance(x, y, eps=0.05)
    assert far["ot_cost"] > near["ot_cost"]
    assert far["w_p_exact"] > 0


def test_barycenter_is_average(measures):
    s = np.asarray(measures["support"])
    out = wasserstein_barycenter([np.asarray(measures["m0"]), np.asarray(measures["m2"])], s)
    b = np.asarray(out["barycenter"])
    c0 = float((np.asarray(measures["m0"]) * s).sum())
    c2 = float((np.asarray(measures["m2"]) * s).sum())
    cb = float((b * s).sum())
    assert abs(cb - 0.5 * (c0 + c2)) < 0.3
    assert abs(b.sum() - 1.0) < 1e-3


def test_ot_plan_shape(measures):
    s = np.asarray(measures["support"])
    p = ot_plan(np.asarray(measures["m0"]), np.asarray(measures["m1"]), _cost(s))
    assert p.shape == (40, 40)
    assert (p >= 0).all()


def test_validation():
    with pytest.raises(ValueError):
        sinkhorn(np.array([0.5, 0.5]), -np.ones(3), np.ones((2, 3)))
    with pytest.raises(ValueError):
        sinkhorn(np.ones(4) / 4, np.ones(4) / 4, np.ones((3, 3)))
    with pytest.raises(ValueError):
        sinkhorn(np.ones(4) / 4, np.ones(4) / 4, np.ones((4, 4)), eps=0.0)
    with pytest.raises(ValueError):
        wasserstein_distance(np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        wasserstein_barycenter([np.ones(4) / 4], np.linspace(0, 1, 4))


def test_determinism(measures):
    s = np.asarray(measures["support"])
    c = _cost(s)
    a = sinkhorn(np.asarray(measures["m0"]), np.asarray(measures["m1"]), c)["ot_cost"]
    b = sinkhorn(np.asarray(measures["m0"]), np.asarray(measures["m1"]), c)["ot_cost"]
    assert a == b


def test_bench_keys():
    out = bench_optimal_transport()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_ot_monotone"] == 1.0
    assert out["synthetic_bary_center_err"] < 0.3
    assert out["synthetic_determinism"] == 1.0
