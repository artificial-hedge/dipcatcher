import numpy as np
import pytest

from quant_fund.models.tmle import bench_tmle, synth_tmle, tmle_ate


def test_tmle_recovers_rd() -> None:
    d = synth_tmle(seed=0)
    rd = float(np.asarray(d["true_rd"])[0])
    out = tmle_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    assert abs(out["ate_tmle"] - rd) < abs(out["naive_diff"] - rd)


def test_tmle_beats_naive_confounding() -> None:
    d = synth_tmle(seed=1)
    out = tmle_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    assert out["naive_diff"] > 0.10
    assert out["ate_tmle"] < 0.10


def test_aipw_close_to_tmle() -> None:
    d = synth_tmle(seed=2)
    out = tmle_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    assert abs(out["ate_aipw"] - out["ate_tmle"]) < 0.02


def test_se_positive() -> None:
    d = synth_tmle(seed=3)
    out = tmle_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    assert 0 < out["se"] < 0.3


def test_zero_effect() -> None:
    d = synth_tmle(effect=0.0, seed=4)
    out = tmle_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    assert abs(out["ate_tmle"]) < 0.08


def test_synth_shapes() -> None:
    d = synth_tmle(n=200, seed=5)
    assert np.asarray(d["y"]).shape == (200,)
    assert np.asarray(d["x"]).shape == (200, 2)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, (n, 2))
    with pytest.raises(ValueError):
        tmle_ate(
            rng.normal(0, 1, 30), rng.integers(0, 2, 30).astype(float), rng.normal(0, 1, (30, 2))
        )
    with pytest.raises(ValueError):
        tmle_ate(rng.normal(0, 1, n), rng.integers(0, 3, n).astype(float), x)
    with pytest.raises(ValueError):
        tmle_ate(rng.normal(0, 1, n) + 2.0, rng.integers(0, 2, n).astype(float), x)
    with pytest.raises(ValueError):
        tmle_ate(rng.uniform(0, 1, n) * np.nan, rng.integers(0, 2, n).astype(float), x)


def test_bench() -> None:
    out = bench_tmle()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
