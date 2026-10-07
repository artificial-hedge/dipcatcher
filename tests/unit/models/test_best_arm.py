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


class _CountingRng:
    """Proxy Generator that counts `.random()` draws (one per arm-pull)."""

    def __init__(self, seed: int) -> None:
        self._g = np.random.default_rng(seed)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return float(self._g.random())


def test_pulls_count_arm_pulls_not_rounds():
    # the "pulls" return must equal the number of arm pulls (rng draws);
    # the old code returned the round counter, undercounting ~2x.
    probs = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    rng = _CountingRng(3)
    _, pulls, _ = successive_elimination(probs, 0.05, rng)
    assert pulls == rng.calls
    rng2 = _CountingRng(4)
    _, lu_pulls, _ = lucb(probs, 0.05, rng2)
    assert lu_pulls == rng2.calls


def test_bench_correct():
    out = bench_best_arm(seed=7)
    assert out["synthetic_se_correct_frac"] == 1.0
    assert out["synthetic_lucb_correct_frac"] == 1.0
