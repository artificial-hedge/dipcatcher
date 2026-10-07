"""Tests for adversarial robustness MLP (models/adversarial_robust.py)."""

import numpy as np

from quant_fund.models.adversarial_robust import Mlp, bench_adversarial_robust


def test_step_uses_pre_update_w2_for_hidden_grad():
    rng = np.random.default_rng(0)
    m = Mlp(seed=3)
    x = rng.normal(0, 1, (64, 8))
    y = (rng.random(64) > 0.5).astype(float)
    w1_old = m.w1.copy()
    w2_old = m.w2.copy()
    h = np.tanh(x @ w1_old + m.b1)
    p = 1.0 / (1.0 + np.exp(-(h @ w2_old + m.b2)))
    dz = p - y
    dh = dz[:, None] * w2_old[None, :] * (1 - h**2)
    gw1 = x.T @ dh / len(y)
    lr = 0.5
    m.step(x, y, lr)
    np.testing.assert_allclose(m.w1, w1_old - lr * gw1, atol=1e-12)


def test_bench_deterministic():
    a = bench_adversarial_robust(seed=5)
    b = bench_adversarial_robust(seed=5)
    assert a == b
