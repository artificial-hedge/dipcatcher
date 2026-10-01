import numpy as np
import pytest

from quant_fund.models.oster_bounds import (
    bench_oster_bounds,
    oster_bounds,
    synth_oster,
)


def test_short_regression_biased_up() -> None:
    d = synth_oster(beta=1.0, gamma=0.8, rho=0.6, seed=0)
    out = oster_bounds(d["y"], d["x"], d["w"])
    assert out["beta_short"] > out["beta_full"]
    assert out["beta_full"] == pytest.approx(1.0, abs=0.15)


def test_bound_improves_on_naive() -> None:
    # Oster's bound moves substantially toward the truth vs the
    # omitted-variable regression
    d = synth_oster(beta=1.0, gamma=0.8, rho=0.6, seed=1)
    out = oster_bounds(d["y"], d["x"], d["w"])
    assert abs(out["beta_star"] - 1.0) < abs(out["beta_short"] - 1.0) * 0.8


def test_delta_star_positive() -> None:
    d = synth_oster(seed=2)
    out = oster_bounds(d["y"], d["x"], d["w"])
    assert out["delta_star"] > 0


def test_no_controls_fallback() -> None:
    d = synth_oster(seed=3)
    out = oster_bounds(d["y"], d["x"], np.zeros((d["y"].size, 0)))
    assert np.isfinite(out["beta_full"])


def test_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        oster_bounds(np.ones(5), np.ones(5), np.zeros((5, 1)))
    with pytest.raises(ValueError):
        oster_bounds(np.ones(60) * np.nan, rng.normal(size=60), np.zeros((60, 1)))
    with pytest.raises(ValueError):
        oster_bounds(rng.normal(size=60), rng.normal(size=60), np.zeros((30, 1)))


def test_deterministic() -> None:
    d = synth_oster(seed=4)
    a = oster_bounds(d["y"], d["x"], d["w"])
    b = oster_bounds(d["y"], d["x"], d["w"])
    assert a["beta_star"] == b["beta_star"]


def test_bench() -> None:
    out = bench_oster_bounds()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
