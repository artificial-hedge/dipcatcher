"""Probes for _opt_synth."""

import numpy as np
import pytest

from quant_fund.models._opt_synth import opt_task


def test_shapes_and_finiteness():
    X, y = opt_task(0, n=50, d=6)
    assert X.shape == (50, 6) and y.shape == (50,)
    assert np.isfinite(X).all() and np.isfinite(y).all()


def test_deterministic():
    a = opt_task(3, n=40)
    b = opt_task(3, n=40)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_implicit_quadratic_psd():
    # y approx 0.5 x'Ax; reconstruct A's spectrum indirectly:
    # y must be mostly positive since A is PSD with eigenvalues > 0
    _, y = opt_task(0, n=500)
    assert (y > -0.5).mean() > 0.9  # noise can push slightly below 0


def test_r_diag_positive_convention_stable():
    # run twice at large d where LAPACK sign flips were historically flaky
    a = opt_task(11, n=32, d=32)[0]
    b = opt_task(11, n=32, d=32)[0]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kw", [{"n": 0}, {"n": -2}, {"d": 0}, {"d": 1}])
def test_hostile_params(kw):
    with pytest.raises(ValueError):
        opt_task(0, **kw)
