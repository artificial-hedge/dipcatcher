"""Probes for _pde_synth (Poisson operator fixture)."""

import numpy as np
import pytest

from quant_fund.models._pde_synth import poisson_pairs, rel_l2


def test_pairs_shapes_and_bc():
    a_tr, u_tr, a_te, u_te = poisson_pairs(seed=0, n_samp=40, n_grid=16, k_max=4)
    assert a_tr.shape == u_tr.shape == (20, 16)
    assert a_te.shape == u_te.shape == (20, 16)
    # Dirichlet BCs: u(x=0)=u(x=1)=0
    np.testing.assert_allclose(u_tr[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(u_tr[:, -1], 0.0, atol=1e-12)


def test_pairs_deterministic():
    a = poisson_pairs(seed=3, n_samp=10)
    b = poisson_pairs(seed=3, n_samp=10)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_operator_relation_u_is_smoother():
    _, u_tr, _, _ = poisson_pairs(seed=0, n_samp=20)
    a_tr = poisson_pairs(seed=0, n_samp=20)[0]
    # u = a smoothed by (kπ)^-2 → high-k energy must shrink
    assert np.std(u_tr) < np.std(a_tr)


@pytest.mark.parametrize("kw", [{"n_samp": 1}, {"n_samp": 0}, {"n_grid": 1}, {"k_max": 0}])
def test_pairs_hostile(kw):
    with pytest.raises(ValueError):
        poisson_pairs(seed=0, **kw)


def test_rel_l2_basics():
    t = np.random.default_rng(0).standard_normal((5, 8))
    assert rel_l2(t, t) == pytest.approx(0.0)
    assert rel_l2(t * 1.1, t) == pytest.approx(0.1, abs=1e-12)


@pytest.mark.parametrize(
    "pred,truth",
    [
        (np.zeros((3, 4)), np.zeros((3, 5))),  # shape mismatch
        (np.zeros((0, 4)), np.zeros((0, 4))),  # empty
        (np.zeros((3, 4)), np.zeros((3, 4))),  # zero-norm truth → NaN laundering
        (np.full((3, 4), np.inf), np.ones((3, 4))),  # non-finite pred
    ],
)
def test_rel_l2_hostile(pred, truth):
    with pytest.raises(ValueError):
        rel_l2(pred, truth)
