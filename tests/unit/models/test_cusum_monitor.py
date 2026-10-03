import numpy as np
import pytest

from quant_fund.models.cusum_monitor import (
    bench_cusum_monitor,
    cusum_monitor,
    synth_break,
)


def test_detects_intercept_break() -> None:
    n = 300
    d = synth_break(n=n, delta=1.5, seed=0)
    out = cusum_monitor(np.asarray(d["y"]), np.asarray(d["x"]))
    assert out["first_cross"] < n + 1
    assert out["first_cross"] > 0.4 * n


def test_silent_under_stability() -> None:
    n = 300
    d = synth_break(n=n, delta=0.0, seed=1)
    out = cusum_monitor(np.asarray(d["y"]), np.asarray(d["x"]))
    assert out["first_cross"] > n
    assert out["max_abs_q"] < 3.0


def test_larger_break_detected_earlier() -> None:
    n = 300
    small = cusum_monitor(np.asarray(synth_break(n=n, delta=1.0, seed=2)["y"]), np.ones(n))
    big = cusum_monitor(np.asarray(synth_break(n=n, delta=2.5, seed=2)["y"]), np.ones(n))
    assert big["first_cross"] <= small["first_cross"]


def test_slope_regressor() -> None:
    rng = np.random.default_rng(3)
    n = 250
    x = rng.normal(0, 1, n)
    y = 1 + 0.5 * x + rng.normal(0, 0.8, n)
    out = cusum_monitor(y, x)
    assert out["max_abs_q"] > 0


def test_synth_shapes() -> None:
    d = synth_break(n=120, seed=4)
    assert np.asarray(d["y"]).shape == (120,)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 100
    with pytest.raises(ValueError):
        cusum_monitor(rng.normal(0, 1, 30), rng.normal(0, 1, 30))
    with pytest.raises(ValueError):
        cusum_monitor(rng.normal(0, 1, n) * np.nan, rng.normal(0, 1, n))
    with pytest.raises(ValueError):
        cusum_monitor(rng.normal(0, 1, n), rng.normal(0, 1, 50))
    with pytest.raises(ValueError):
        cusum_monitor(rng.normal(0, 1, n), rng.normal(0, 1, n), m=2)
    with pytest.raises(ValueError):
        cusum_monitor(rng.normal(0, 1, n), rng.normal(0, 1, n), alpha=0.8)


def test_bench() -> None:
    out = bench_cusum_monitor()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
