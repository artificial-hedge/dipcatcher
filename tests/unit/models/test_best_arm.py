import numpy as np

from quant_fund.models.best_arm import bench_best_arm, lucb, successive_elimination


def test_se_finds_best():
    rng = np.random.default_rng(0)
    probs = np.array([0.7, 0.4, 0.3])
    arm, pulls, ok = successive_elimination(probs, 0.05, rng)
    assert ok and arm == 0 and pulls > 0


def test_lucb_finds_best():
    rng = np.random.default_rng(1)
    probs = np.array([0.7, 0.4, 0.3])
    arm, pulls, ok = lucb(probs, 0.05, rng)
    assert ok and arm == 0 and pulls > 0


def test_bench_correct():
    out = bench_best_arm(seed=7)
    assert out["synthetic_se_correct_frac"] == 1.0
    assert out["synthetic_lucb_correct_frac"] == 1.0
