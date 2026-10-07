"""Adversarial probes for _mt_core (torch-gated).

Honesty defects probed:
- vacuous iters <= 0 returning fabricated accuracy → must raise
- training loop leaking global torch RNG state (canonical-default drift
  vs siblings that fork_rng) → must leave torch RNG untouched
- combine() returning non-finite / wrong-shape gradient → must raise
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quant_fund.models._mt_core import train_mtl  # noqa: E402


def _id_combine(g1, g2):
    return g1 + g2


def test_vacuous_iters_rejected():
    for bad in (0, -3):
        with pytest.raises(ValueError):
            train_mtl(_id_combine, seed=0, iters=bad)


def test_rng_isolated_from_global_state():
    torch.manual_seed(12345)
    torch.randn(3)  # advance global state
    before = torch.random.get_rng_state()
    train_mtl(_id_combine, seed=3, iters=2)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_deterministic_same_seed():
    a = train_mtl(_id_combine, seed=7, iters=15)
    b = train_mtl(_id_combine, seed=7, iters=15)
    assert a == b
    assert all(0.0 <= v <= 1.0 for v in a)


def test_combine_output_validated():
    def bad_shape(g1, g2):
        return torch.zeros(2)  # wrong length vs trunk params

    def bad_finite(g1, g2):
        return torch.full(g1.shape, np.inf)

    for fn in (bad_shape, bad_finite):
        with pytest.raises(ValueError):
            train_mtl(fn, seed=0, iters=1, naive=False)


def test_naive_path_runs_and_scores():
    lo, mean = train_mtl(_id_combine, seed=5, iters=20, naive=True)
    assert 0.0 <= lo <= mean <= 1.0
