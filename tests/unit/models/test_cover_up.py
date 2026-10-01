import numpy as np
import pytest

from quant_fund.models.cover_up import (
    bench_cover_up,
    synth_portfolio,
    universal_portfolio,
)


def test_tracks_winner() -> None:
    d = synth_portfolio(seed=0)
    out = universal_portfolio(np.asarray(d["returns"]))
    assert out["w_max"] > 0.6


def test_regret_bounded() -> None:
    d = synth_portfolio(seed=1)
    out = universal_portfolio(np.asarray(d["returns"]))
    assert out["log_regret"] < 1.5
    assert out["log_regret"] > 0


def test_flat_tape_balanced() -> None:
    rng = np.random.default_rng(2)
    rr = np.clip(1.0 + rng.normal(0, 0.01, (200, 2)), 0.8, 1.2)
    out = universal_portfolio(rr)
    assert out["w_max"] < 0.75


def test_wealth_positive() -> None:
    d = synth_portfolio(seed=3)
    out = universal_portfolio(np.asarray(d["returns"]))
    assert out["final_wealth"] > 1.0


def test_three_assets() -> None:
    rng = np.random.default_rng(4)
    rr = np.clip(1.0 + rng.normal([0.0, 0.005, 0.0], 0.02, (150, 3)), 0.5, 1.5)
    out = universal_portfolio(rr, grid=8)
    assert np.isfinite(out["final_wealth"])


def test_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        universal_portfolio(rng.uniform(0.9, 1.1, 5)[:, None])
    with pytest.raises(ValueError):
        universal_portfolio(np.ones((50, 3)) * -1)
    with pytest.raises(ValueError):
        universal_portfolio(np.ones((50, 3)) * np.nan)
    with pytest.raises(ValueError):
        universal_portfolio(np.ones((50, 3)), alpha=9.0)


def test_bench() -> None:
    out = bench_cover_up()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
