"""Tests for crossq — batch-norm critic CrossQ bench."""

from __future__ import annotations

import numpy as np

from quant_fund.models.crossq import bench_crossq


def test_random_reward_is_measured_not_constant() -> None:
    # the baseline used to be a hardcoded -0.2 — it must be an actual
    # uniform-random policy simulation on the same env.
    from quant_fund.models.crossq import _eval_random

    v = _eval_random(np.random.default_rng(919))
    assert np.isfinite(v)
    assert v != -0.2
    # deterministic given the seed
    assert _eval_random(np.random.default_rng(919)) == v


def test_random_reward_varies_with_env_seed() -> None:
    from quant_fund.models.crossq import _eval_random

    vals = {_eval_random(np.random.default_rng(s)) for s in (1, 2, 3, 4, 5)}
    assert len(vals) > 1  # a measured baseline responds to the env draw


def test_bench_crossq_smoke() -> None:
    out = bench_crossq(steps=400)
    for key, val in out.items():
        assert key.startswith("synthetic_"), key
        assert np.isfinite(val), key
