"""Probes: the ATE standard error must use the delta method over the
(ca, caw) block — not just Var(ca)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.proximal_causal import _ols, proximal_ate


def _run(seed: int = 3, n: int = 2000):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    x = rng.normal(size=n)
    z = u + rng.normal(scale=0.5, size=n)
    w = 0.7 * u + rng.normal(scale=0.4, size=n)
    g = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * u + 0.3 * x)))
    a = (rng.uniform(size=n) < g).astype(float)
    # heterogeneous effect: A*W interaction so caw != 0
    y = 1.0 * a + 0.4 * a * w + 0.8 * u + 0.5 * x + rng.normal(scale=0.5, size=n)
    return proximal_ate(y, a, z, w, x)


def test_se_uses_delta_method() -> None:
    out = _run()
    assert out["se"] > 0 and np.isfinite(out["se"])


def test_se_reflects_interaction_uncertainty() -> None:
    """Closed-form: se_ate == sqrt(s2 * g' (Xhat'Xhat)^-1 g)."""
    rng = np.random.default_rng(3)
    n = 2000
    u = rng.normal(size=n)
    x = rng.normal(size=n)
    z = u + rng.normal(scale=0.5, size=n)
    w = 0.7 * u + rng.normal(scale=0.4, size=n)
    g0 = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * u + 0.3 * x)))
    a = (rng.uniform(size=n) < g0).astype(float)
    y = 1.0 * a + 0.4 * a * w + 0.8 * u + 0.5 * x + rng.normal(scale=0.5, size=n)
    out = proximal_ate(y, a, z, w, x)
    # reconstruct the internals to compute the reference se
    tt, zz, ww = a, z[:, None], w[:, None]
    xx = x[:, None]
    one = np.ones((n, 1))
    aw = tt[:, None] * ww
    endo = np.column_stack([ww, aw])
    exo = np.column_stack([one, tt, xx])
    instr = np.column_stack([one, tt, xx, zz, tt[:, None] * zz])
    endo_hat = np.column_stack([instr @ _ols(endo[:, j], instr) for j in range(endo.shape[1])])
    reg2 = np.column_stack([exo, endo_hat])
    resid = y - np.column_stack([exo, endo]) @ _ols(y, reg2)
    s2 = float(resid @ resid) / (n - reg2.shape[1])
    xtx_inv = np.linalg.pinv(reg2.T @ reg2)
    g = np.zeros(reg2.shape[1])
    g[1] = 1.0
    g[exo.shape[1] + 1 :] = ww.mean(axis=0)
    ref = float(np.sqrt(s2 * float(g @ xtx_inv @ g)))
    assert out["se"] == pytest.approx(ref, rel=1e-9)
    # and it must exceed the naive Var(ca)-only form when caw matters
    naive_se = float(np.sqrt(s2 * xtx_inv[1, 1]))
    assert out["se"] != pytest.approx(naive_se, rel=1e-6)
