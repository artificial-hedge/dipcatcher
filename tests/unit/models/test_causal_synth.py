"""Unit tests for quant_fund.models._causal_synth."""

from __future__ import annotations

import numpy as np

from quant_fund.models._causal_synth import synth_iv, synth_observational


def test_observational_deterministic_and_shapes() -> None:
    a = synth_observational(seed=4, n=200)
    b = synth_observational(seed=4, n=200)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))
    x, t, y, tau, e = a
    assert x.shape == (200, 5)
    assert set(np.unique(t)) <= {0, 1}
    # tau = 1 + x0
    assert np.allclose(tau, 1 + x[:, 0])
    # propensity in (0,1)
    assert np.all((e > 0) & (e < 1))


def test_observational_true_ate() -> None:
    x, _t, _y, tau, _e = synth_observational(seed=0, n=4000)
    # ATE = E[1 + x0] = 1 since x0 ~ N(0,1)
    assert abs(tau.mean() - 1.0) < 0.05


def test_iv_deterministic_and_effect() -> None:
    z1, t1, y1, fx1 = synth_iv(seed=7, n=300)
    z2, t2, y2, fx2 = synth_iv(seed=7, n=300)
    assert np.array_equal(z1, z2) and np.array_equal(t1, t2) and np.array_equal(y1, y2)
    assert np.allclose(fx1, 1.5)
    assert np.array_equal(fx1, fx2)
