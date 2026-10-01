import numpy as np
import pytest

from quant_fund.models.hausman_tests import (
    bench_hausman_tests,
    dwh_test,
    hausman_fe_re,
    synth_endo,
    synth_fe_panel,
)


def test_fe_re_rejects_correlated_effect() -> None:
    d = synth_fe_panel(corr_alpha_x=0.8, seed=0)
    out = hausman_fe_re(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["firm"]))
    assert out["p"] < 0.05
    assert out["beta_fe"] < out["beta_re"]


def test_fe_re_accepts_uncorrelated() -> None:
    d = synth_fe_panel(corr_alpha_x=0.0, seed=1)
    out = hausman_fe_re(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["firm"]))
    assert out["p"] > 0.05


def test_fe_recovers_beta() -> None:
    d = synth_fe_panel(corr_alpha_x=0.8, beta=1.0, seed=2)
    out = hausman_fe_re(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["firm"]))
    assert abs(out["beta_fe"] - 1.0) < 0.15


def test_dwh_rejects_endogeneity() -> None:
    d = synth_endo(rho=0.8, seed=3)
    out = dwh_test(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert out["p"] < 0.05


def test_dwh_accepts_exogeneity() -> None:
    d = synth_endo(rho=0.0, seed=4)
    out = dwh_test(np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"]))
    assert out["p"] > 0.05


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 120
    firm = np.repeat(np.arange(30), 4).astype(float)
    with pytest.raises(ValueError):
        hausman_fe_re(rng.normal(0, 1, 30), rng.normal(0, 1, 30), firm[:30])
    with pytest.raises(ValueError):
        hausman_fe_re(rng.normal(0, 1, n) * np.nan, rng.normal(0, 1, n), firm)
    with pytest.raises(ValueError):
        dwh_test(rng.normal(0, 1, 30), rng.normal(0, 1, 30), rng.normal(0, 1, 30))
    with pytest.raises(ValueError):
        dwh_test(
            rng.normal(0, 1, n),
            rng.normal(0, 1, n),
            rng.normal(0, 1, 50),
        )


def test_bench() -> None:
    out = bench_hausman_tests()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
