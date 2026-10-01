import numpy as np
import pytest

from quant_fund.models.stambaugh import (
    bench_stambaugh,
    stambaugh_pred,
    synth_pred,
)


def test_stambaugh_bias_correction() -> None:
    d = synth_pred(beta=0.05, rho=0.95, corr=-0.8, seed=7)
    out = stambaugh_pred(d["y"], d["x"])
    assert abs(out["beta_corrected"] - 0.05) < abs(out["beta_ols"] - 0.05)


def test_stambaugh_rho_recovered() -> None:
    d = synth_pred(rho=0.9, seed=3)
    out = stambaugh_pred(d["y"], d["x"])
    assert abs(out["rho_hat"] - 0.9) < 0.1


def test_stambaugh_q_ci_contains_beta() -> None:
    d = synth_pred(beta=0.05, seed=4)
    out = stambaugh_pred(d["y"], d["x"])
    assert out["beta_q_lo"] <= 0.05 <= out["beta_q_hi"]


def test_stambaugh_uncorr_small_bias() -> None:
    d = synth_pred(rho=0.5, corr=0.0, seed=5)
    out = stambaugh_pred(d["y"], d["x"])
    assert abs(out["bias"]) < 0.02


def test_stambaugh_validation() -> None:
    with pytest.raises(ValueError):
        stambaugh_pred(np.ones(100), np.ones(100))
    with pytest.raises(ValueError):
        stambaugh_pred(np.zeros(30), np.zeros(30))
    d = synth_pred(seed=6)
    with pytest.raises(ValueError):
        stambaugh_pred(d["y"][:-1], d["x"])


def test_stambaugh_deterministic() -> None:
    d = synth_pred(seed=7)
    a = stambaugh_pred(d["y"], d["x"])
    b = stambaugh_pred(d["y"], d["x"])
    assert a["beta_corrected"] == b["beta_corrected"]


def test_bench_stambaugh() -> None:
    out = bench_stambaugh()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
