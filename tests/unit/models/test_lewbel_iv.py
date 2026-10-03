import numpy as np
import pytest

from quant_fund.models.lewbel_iv import bench_lewbel_iv, lewbel_iv, synth_lewbel


def test_lewbel_corrects_endogeneity() -> None:
    d = synth_lewbel(seed=0)
    out = lewbel_iv(np.asarray(d["y"]), np.asarray(d["x_endo"]), np.asarray(d["x_exo"]))
    assert abs(out["beta_lewbel"] - 1.0) < abs(out["beta_ols"] - 1.0)
    assert abs(out["beta_lewbel"] - 1.0) < 0.15


def test_ols_biased() -> None:
    d = synth_lewbel(seed=1, rho=0.8)
    out = lewbel_iv(np.asarray(d["y"]), np.asarray(d["x_endo"]), np.asarray(d["x_exo"]))
    assert out["beta_ols"] > 1.15


def test_first_stage_strong() -> None:
    d = synth_lewbel(seed=2)
    out = lewbel_iv(np.asarray(d["y"]), np.asarray(d["x_endo"]), np.asarray(d["x_exo"]))
    assert out["f_first_stage"] > 10


def test_zero_beta() -> None:
    d = synth_lewbel(beta=0.0, seed=3)
    out = lewbel_iv(np.asarray(d["y"]), np.asarray(d["x_endo"]), np.asarray(d["x_exo"]))
    assert abs(out["beta_lewbel"]) < 0.2


def test_synth_shapes() -> None:
    d = synth_lewbel(n=300, seed=4)
    assert np.asarray(d["y"]).shape == (300,)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 200
    with pytest.raises(ValueError):
        lewbel_iv(rng.normal(0, 1, 30), rng.normal(0, 1, 30), rng.normal(0, 1, 30))
    with pytest.raises(ValueError):
        lewbel_iv(rng.normal(0, 1, n) * np.nan, rng.normal(0, 1, n), rng.normal(0, 1, n))
    with pytest.raises(ValueError):
        lewbel_iv(rng.normal(0, 1, n), rng.normal(0, 1, (n, 2)), rng.normal(0, 1, n))
    with pytest.raises(ValueError):
        lewbel_iv(rng.normal(0, 1, n), rng.normal(0, 1, 50), rng.normal(0, 1, 50))


def test_bench() -> None:
    out = bench_lewbel_iv()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
