"""Unit tests for quant_fund.models._it_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._it_synth import (
    dep_data,
    med_bw,
    rbf,
    true_mi_gauss,
    twosample,
)


def test_dep_data_kinds() -> None:
    xd, yd = dep_data(0, n=4000, kind="dep")
    xi, yi = dep_data(0, n=4000, kind="indep")
    xn, yn = dep_data(0, n=4000, kind="nonlin")
    assert abs(np.corrcoef(xd, yd)[0, 1]) > 0.9  # strong linear dependence
    assert abs(np.corrcoef(xi, yi)[0, 1]) < 0.05  # independent
    # sin(2x) keeps a partial linear component (~-0.38): weaker than dep,
    # clearly above independent noise — dependence is real but nonlinear
    cn = abs(np.corrcoef(xn, yn)[0, 1])
    assert 0.2 < cn < 0.6


def test_dep_data_rejects_unknown_kind() -> None:
    # typo 'depp' used to fall into the else branch and return INDEPENDENT
    # data labeled as the requested kind — dead-kwarg mislabel
    with pytest.raises(ValueError, match="kind"):
        dep_data(0, kind="depp")
    with pytest.raises(ValueError, match="kind"):
        dep_data(0, kind="dependent")
    with pytest.raises(ValueError, match="n"):
        dep_data(0, n=0)


def test_dep_data_determinism() -> None:
    x1, y1 = dep_data(5, n=50)
    x2, y2 = dep_data(5, n=50)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_twosample_shift() -> None:
    x, y = twosample(0, n=3000, shift=1.0)
    assert y.mean() - x.mean() == pytest.approx(1.0, abs=0.1)
    with pytest.raises(ValueError, match="n"):
        twosample(0, n=0)


def test_rbf_kernel_range() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 2))
    K = rbf(x, x, bw=1.0)
    assert K.shape == (10, 10)
    np.testing.assert_allclose(np.diag(K), np.ones(10))
    assert (K <= 1.0).all() and (K > 0).all()


def test_rbf_rejects_bad_bandwidth() -> None:
    x = np.ones((3, 2))
    with pytest.raises(ValueError, match="bandwidth"):
        rbf(x, x, bw=0.0)
    with pytest.raises(ValueError, match="bandwidth"):
        rbf(x, x, bw=-1.0)


def test_med_bw_reasonable() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2))
    Y = rng.standard_normal((50, 2))
    bw = med_bw(X, Y)
    assert 0.5 < bw < 10.0


def test_med_bw_rejects_constant_data() -> None:
    # identical points -> d[d>0] empty -> median(nan) returned as bandwidth
    X = np.zeros((4, 2))
    with pytest.raises(ValueError, match="median"):
        med_bw(X, X)


def test_true_mi_gauss_values() -> None:
    assert true_mi_gauss(0.0) == pytest.approx(0.0, abs=1e-12)
    assert true_mi_gauss(0.5) == pytest.approx(-0.5 * np.log(0.75))
    assert true_mi_gauss(0.9) > true_mi_gauss(0.5)


def test_true_mi_gauss_rejects_perfect_corr() -> None:
    # |corr|=1 means infinite MI; clamping to -0.5*log(1e-9) ≈ 10.36
    # laundered a divergent quantity into a plausible finite number
    with pytest.raises(ValueError, match="corr"):
        true_mi_gauss(1.0)
    with pytest.raises(ValueError, match="corr"):
        true_mi_gauss(-1.0)
    with pytest.raises(ValueError, match="corr"):
        true_mi_gauss(1.3)
