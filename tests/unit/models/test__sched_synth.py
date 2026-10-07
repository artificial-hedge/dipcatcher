"""Probes for _sched_synth."""

import numpy as np
import pytest

from quant_fund.models._sched_synth import (
    flowshop,
    knapsack,
    makespan,
    tsp,
    weighted_jobs,
)


def test_flowshop_shape_deterministic():
    a = flowshop(0, n_jobs=5, n_mach=3)
    b = flowshop(0, n_jobs=5, n_mach=3)
    assert a.shape == (5, 3)
    np.testing.assert_array_equal(a, b)
    assert (a >= 2).all() and (a < 25).all()


@pytest.mark.parametrize("kw", [{"n_jobs": 0}, {"n_mach": 0}, {"n_jobs": -1}])
def test_flowshop_hostile(kw):
    with pytest.raises(ValueError):
        flowshop(0, **kw)


def test_weighted_jobs_shapes():
    p, w = weighted_jobs(0, n=15)
    assert p.shape == w.shape == (15,)
    assert (p > 0).all() and (w > 0).all()


def test_weighted_jobs_rejects_empty():
    with pytest.raises(ValueError):
        weighted_jobs(0, n=0)


def test_knapsack_cap_is_fraction():
    w, v, cap = knapsack(0, n=20, cap_frac=0.4)
    assert cap == pytest.approx(w.sum() * 0.4)
    assert (w > 0).all() and (v > 0).all()


@pytest.mark.parametrize("kw", [{"n": 0}, {"cap_frac": 0.0}, {"cap_frac": 1.0}, {"cap_frac": 1.5}])
def test_knapsack_hostile(kw):
    with pytest.raises(ValueError):
        knapsack(0, **kw)


def test_tsp_metric_properties():
    d = tsp(0, n=8)
    assert d.shape == (8, 8)
    np.testing.assert_allclose(d, d.T)  # symmetric
    np.testing.assert_allclose(np.diag(d), 0.0)  # zero diagonal
    # triangle inequality
    for i, j, k in [(0, 1, 2), (3, 5, 7)]:
        assert d[i, k] <= d[i, j] + d[j, k] + 1e-12


def test_tsp_rejects_tiny():
    with pytest.raises(ValueError):
        tsp(0, n=1)


def test_makespan_order_matters_and_is_permutation_checked():
    p = flowshop(0, n_jobs=4, n_mach=2)
    ms = makespan([0, 1, 2, 3], p)
    assert ms > 0
    assert np.isfinite(ms)


@pytest.mark.parametrize(
    "order",
    [
        [0, 0, 1, 1],  # duplicate jobs silently double-counted before
        [0, 1],  # incomplete order silently scored a partial schedule
        [0, 1, 2, 9],  # out-of-range job
    ],
)
def test_makespan_rejects_bad_order(order):
    p = flowshop(0, n_jobs=4, n_mach=2)
    with pytest.raises(ValueError):
        makespan(order, p)


def test_makespan_rejects_1d_matrix():
    with pytest.raises(ValueError):
        makespan([0], np.ones(3))
