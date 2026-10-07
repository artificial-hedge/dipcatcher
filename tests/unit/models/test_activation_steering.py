"""Tests for activation steering bench (models/activation_steering.py)."""

import numpy as np

import quant_fund.models.activation_steering as m


def test_steering_vector_uses_train_half_only(monkeypatch):
    # Train half: the target feature separates cleanly along e0 (±4).
    # Eval half: the feature direction tilts into e1. If v is fit on ALL data
    # the eval-half tilt leaks into v, the steered direction is diluted, and
    # steering fails to flip the eval probe (flip ≈ 0). Fit on the train
    # half only, steering subtracts ≈ e0 and flips half the eval points.
    n, d = 200, 4
    rng = np.random.default_rng(0)
    s = np.zeros((n, 3))
    s[:50, 0] = 1.0
    s[100:150, 0] = 1.0
    e0 = np.zeros(d)
    e0[0] = 1.0
    e1 = np.zeros(d)
    e1[1] = 1.0
    a = 0.02 * rng.standard_normal((n, d))
    a[:50] += 4 * e0
    a[50:100] -= 4 * e0
    a[100:150] += 1.6 * e0 + 6 * e1
    a[150:] -= 1.6 * e0 + 6 * e1
    monkeypatch.setattr(m, "feature_directions", lambda rng_: np.zeros(d))
    monkeypatch.setattr(m, "synth_activations", lambda *a_, **k: (a, s))
    out = m.bench_activation_steering(seed=0, n=n, target=0)
    assert out["synthetic_steer_flip_rate"] > 0.3
