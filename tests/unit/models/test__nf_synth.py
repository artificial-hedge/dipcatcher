"""Probes for _nf_synth."""

import numpy as np
import pytest

from quant_fund.models._nf_synth import gauss_nll, pinwheel, two_moons


def test_pinwheel_shape_deterministic():
    a = pinwheel(0, n=100)
    b = pinwheel(0, n=100)
    assert a.shape == (100, 2)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all()


@pytest.mark.parametrize("kw", [{"n": 0}, {"arms": 1}, {"d": 3}])
def test_pinwheel_hostile(kw):
    with pytest.raises(ValueError):
        pinwheel(0, **kw)


def test_two_moons_shape_deterministic():
    a = two_moons(1, n=51)
    assert a.shape == (51, 2)
    np.testing.assert_array_equal(a, two_moons(1, n=51))


def test_two_moons_n_guard():
    with pytest.raises(ValueError):
        two_moons(0, n=0)


def test_gauss_nll_finite_and_lower_for_inlier():
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((200, 2))
    Xte = rng.standard_normal((50, 2))
    nll = gauss_nll(Xtr, Xte)
    assert np.isfinite(nll)
    # near-mean points should score better than far outliers
    assert gauss_nll(Xtr, np.zeros((5, 2))) < gauss_nll(Xtr, np.full((5, 2), 9.0))


@pytest.mark.parametrize(
    "Xtr,Xte",
    [
        (np.zeros((1, 2)), np.zeros((3, 2))),  # singleton train -> degenerate cov
        (np.zeros((10, 2)), np.zeros((0, 2))),  # empty eval
        (np.zeros((10, 2)), np.zeros((3, 3))),  # dim mismatch
        (np.zeros(5), np.zeros((3, 2))),  # 1-D
        (np.full((10, 2), np.nan), np.zeros((3, 2))),  # non-finite
    ],
)
def test_gauss_nll_hostile(Xtr, Xte):
    with pytest.raises(ValueError):
        gauss_nll(Xtr, Xte)
