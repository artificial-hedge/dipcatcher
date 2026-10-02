import numpy as np

from quant_fund.models.adversarial_bandits import (
    bench_adversarial_bandits,
    exp3,
    hedge,
)


def test_exp3_nonnegative():
    rng = np.random.default_rng(0)
    tab = rng.random((200, 4))
    assert exp3(tab, rng) >= 0.0


def test_hedge_below_uniform():
    tab = np.zeros((300, 3))
    tab[:, 0] = 0.9
    tab[:, 1:] = 0.2
    h = hedge(tab)
    uniform = float(np.sum(tab.max(axis=1) - tab.mean(axis=1)))
    assert h < uniform


def test_bench_exp3_beats_uniform():
    out = bench_adversarial_bandits(seed=6)
    assert out["synthetic_exp3_beats_uniform"] == 1.0
    assert out["synthetic_exp3_regret"] < out["synthetic_uniform_regret"]
