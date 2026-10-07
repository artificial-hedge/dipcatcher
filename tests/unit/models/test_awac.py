"""Tests for models/awac.py — the random baseline must be measured."""

from __future__ import annotations

import numpy as np


def test_random_baseline_is_measured(monkeypatch) -> None:
    """synthetic_awac_random_reward was a hardcoded -0.2; it must come
    from a real random-policy roll-out on the same env/step budget."""
    import quant_fund.models.awac as awac

    def stub_env(state, a, rng):
        return np.zeros(4), 0.77

    monkeypatch.setattr(awac, "_env_step", stub_env)
    out = awac.bench_awac(seed=0, steps=10)
    assert abs(out["synthetic_awac_random_reward"] - 0.77) < 1e-12
    assert abs(out["synthetic_awac_mean_reward"] - 0.77) < 1e-12
