"""Probes: the Huber-White meat on a weighted fit needs w_i^2 * resid^2.
Constant weights c must reproduce the unweighted HC0 SE exactly
(buggy w^1 meat scales the SE by 1/sqrt(c))."""

from __future__ import annotations

import numpy as np
import pytest

import quant_fund.models.msm_causal as mc


def _data(seed: int = 7, n: int = 600):
    rng = np.random.default_rng(seed)
    a1 = rng.normal(size=n)
    a2 = rng.normal(size=n)
    y = 0.5 * (a1 + a2) + rng.normal(size=n)
    return y, a1, a2


def _hc0_se(y: np.ndarray, dose: np.ndarray) -> float:
    x = np.column_stack([np.ones(y.size), dose])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    r = y - x @ beta
    meat = x.T @ (r**2)[:, None] @ x if False else x.T @ ((r**2)[:, None] * x)
    bread = x.T @ x
    cov = np.linalg.solve(bread, meat @ np.linalg.inv(bread))
    return float(np.sqrt(cov[1, 1]))


def test_constant_weights_se_invariant(monkeypatch) -> None:
    y, a1, a2 = _data()
    n = y.size
    lc = np.zeros((n, 1))
    b = np.zeros((n, 1))
    monkeypatch.setattr(mc, "stabilized_weights", lambda *a, **k: np.full(n, 3.0))
    out = mc.msm_cumulative_effect(y, a1, lc, a2, lc, b)
    assert out["psi1_se"] == pytest.approx(_hc0_se(y, a1 + a2), rel=1e-9)


def test_sandwich_matches_closed_form() -> None:
    # With the real weights, verify se equals the reference w^2 sandwich.
    rng = np.random.default_rng(11)
    n = 800
    base = rng.normal(size=n)
    l1 = base + rng.normal(scale=0.5, size=n)
    a1 = 0.5 * l1 + rng.normal(scale=0.9, size=n)
    l2 = l1 + 0.7 * a1 + rng.normal(scale=0.6, size=n)
    a2 = 0.5 * l2 + 0.2 * a1 + rng.normal(scale=0.9, size=n)
    y = 0.5 * (a1 + a2) + 1.3 * l1 + rng.normal(scale=0.9, size=n)
    out = mc.msm_cumulative_effect(y, a1, l1, a2, l2, base)
    sw = np.asarray(out["weights"])
    dose = a1 + a2
    x = np.column_stack([np.ones(n), dose])
    w_sqrt = np.sqrt(sw)
    beta = np.linalg.lstsq(x * w_sqrt[:, None], y * w_sqrt, rcond=None)[0]
    r = y - x @ beta
    meat = x.T @ ((sw**2 * r**2)[:, None] * x)
    bread = x.T @ (sw[:, None] * x)
    cov = np.linalg.solve(bread, meat @ np.linalg.inv(bread))
    assert out["psi1_se"] == pytest.approx(float(np.sqrt(cov[1, 1])), rel=1e-9)


def test_nonfinite_inputs_fail_closed() -> None:
    y, a1, a2 = _data()
    y[10] = np.nan
    lc = np.zeros((y.size, 1))
    b = np.zeros((y.size, 1))
    with pytest.raises(ValueError, match="non-finite"):
        mc.msm_cumulative_effect(y, a1, lc, a2, lc, b)
