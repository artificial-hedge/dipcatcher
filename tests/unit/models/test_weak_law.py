import numpy as np
import pytest

from quant_fund.models import weak_law as wl


def test_bench_weak_law_passes() -> None:
    out = wl.bench_weak_law(seed=0)
    assert out["synthetic_weak_law"] == 1.0


def test_bench_checks_actually_exercise_the_estimator() -> None:
    """Every check must consult sample_mean_error: the old bench had 3
    literal-True entries and scored 0.8 even with a broken estimator."""
    monkey = pytest.MonkeyPatch()
    monkey.setattr(wl, "sample_mean_error", lambda n, rng: 5.0)
    out = wl.bench_weak_law(seed=0)
    monkey.undo()
    assert out["synthetic_weak_law"] < 0.5


def test_sample_mean_error_scales_down() -> None:
    rng = np.random.default_rng(1)
    e100 = np.mean([wl.sample_mean_error(100, rng) for _ in range(30)])
    e40000 = np.mean([wl.sample_mean_error(40000, rng) for _ in range(30)])
    assert e40000 < e100 / 5
