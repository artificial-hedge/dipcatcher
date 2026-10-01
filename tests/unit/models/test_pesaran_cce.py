import numpy as np
import pytest

from quant_fund.models.pesaran_cce import (
    bench_pesaran_cce,
    cce_mean_group,
    cce_slopes,
    synth_cce_panel,
)


def test_ccemg_recovers_beta() -> None:
    d = synth_cce_panel(beta=1.0, seed=0)
    b = cce_mean_group(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["i_idx"]))
    assert abs(b - 1.0) < 0.15


def test_ols_is_biased() -> None:
    d = synth_cce_panel(beta=1.0, seed=1)
    b_ols = float(np.polyfit(np.asarray(d["x"]), np.asarray(d["y"]), 1)[0])
    assert abs(b_ols - 1.0) > 0.3


def test_slopes_shape() -> None:
    d = synth_cce_panel(n_per=20, t=40, seed=2)
    s = cce_slopes(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["i_idx"]))
    assert s.shape == (20,)


def test_no_factor_unbiased() -> None:
    d = synth_cce_panel(beta=1.0, gamma_x=0.0, gamma_y=0.0, seed=3)
    b = cce_mean_group(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["i_idx"]))
    assert abs(b - 1.0) < 0.2


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 100
    with pytest.raises(ValueError):
        cce_slopes(rng.normal(0, 1, 10), rng.normal(0, 1, 10), np.zeros(10))
    with pytest.raises(ValueError):
        cce_slopes(
            rng.normal(0, 1, n) * np.nan,
            rng.normal(0, 1, n),
            np.zeros(n),
        )
    # unbalanced
    ii = np.array([0] * 60 + [1] * 40, dtype=float)
    with pytest.raises(ValueError):
        cce_slopes(rng.normal(0, 1, 100), rng.normal(0, 1, 100), ii)


def test_bench() -> None:
    out = bench_pesaran_cce()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
