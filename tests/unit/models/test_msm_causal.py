"""Tests for msm_causal — Robins MSM."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.msm_causal import (
    bench_msm_causal,
    msm_cumulative_effect,
    stabilized_weights,
)


def _dgp(seed: int = 0, n: int = 2500, psi: float = 0.5):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    l1 = 0.8 * base + rng.normal(scale=0.5, size=n)
    a1 = 0.5 * l1 + rng.normal(scale=0.9, size=n)
    l2 = 0.6 * l1 + 0.7 * a1 + rng.normal(scale=0.6, size=n)
    a2 = 0.5 * l2 + 0.2 * a1 + rng.normal(scale=0.9, size=n)
    y = psi * (a1 + a2) + 1.3 * l1 + rng.normal(scale=0.9, size=n)
    return y, a1, l1, a2, l2, base, psi


def test_weights_finite_positive():
    y, a1, l1, a2, l2, base, _ = _dgp()
    sw = stabilized_weights(a1, l1, a2, l2, base)
    assert np.all(np.isfinite(sw))
    assert np.all(sw > 0)
    assert sw.mean() == pytest.approx(1.0, abs=0.6)


def test_msm_recovers_psi():
    y, a1, l1, a2, l2, base, psi = _dgp()
    out = msm_cumulative_effect(y, a1, l1, a2, l2, base)
    assert abs(out["psi1"] - psi) < 0.2


def test_msm_better_than_naive():
    y, a1, l1, a2, l2, base, psi = _dgp()
    out = msm_cumulative_effect(y, a1, l1, a2, l2, base)
    assert abs(out["psi1"] - psi) <= abs(out["naive_beta1"] - psi)


def test_unconfounded_still_works():
    rng = np.random.default_rng(5)
    n = 2000
    base = rng.normal(size=n)
    l1 = rng.normal(size=n)
    a1 = (rng.normal(size=n) > 0).astype(np.float64)  # randomized
    l2 = rng.normal(size=n)
    a2 = (rng.normal(size=n) > 0).astype(np.float64)
    y = 0.7 * (a1 + a2) + rng.normal(size=n)
    out = msm_cumulative_effect(y, a1, l1, a2, l2, base)
    assert abs(out["psi1"] - 0.7) < 0.2


def test_fail_closed_shape():
    with pytest.raises(ValueError):
        msm_cumulative_effect(
            np.ones(10),
            np.ones(9),
            np.ones((10, 1)),
            np.ones(10),
            np.ones((10, 1)),
            np.ones((10, 1)),
        )


def test_bench():
    out = bench_msm_causal()
    assert out["score"] == 1.0
