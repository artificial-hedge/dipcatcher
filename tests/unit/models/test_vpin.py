import numpy as np
import pytest

from quant_fund.models.vpin import bench_vpin, synth_vpin, vpin


def test_detects_informed_bursts() -> None:
    d = synth_vpin(seed=0)
    out = vpin(np.asarray(d["price"]), np.asarray(d["volume"]))
    assert out["vpin_max"] > 0.15


def test_quiet_tape_low() -> None:
    rng = np.random.default_rng(1)
    p = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    v = rng.uniform(0.5, 1.5, 3000)
    out = vpin(p, v)
    assert out["vpin_max"] < 0.15


def test_buckets_filled() -> None:
    d = synth_vpin(seed=2)
    out = vpin(np.asarray(d["price"]), np.asarray(d["volume"]), n_buckets=40)
    assert out["n_filled"] >= 39


def test_imbalance_nonnegative() -> None:
    d = synth_vpin(seed=3)
    out = vpin(np.asarray(d["price"]), np.asarray(d["volume"]))
    assert out["vpin_mean"] >= 0
    assert out["vpin_max"] <= 1.0


def test_synth_shapes() -> None:
    d = synth_vpin(n=1000, seed=4)
    assert np.asarray(d["price"]).shape == (1000,)
    assert np.asarray(d["volume"]).shape == (1000,)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        vpin(rng.uniform(90, 110, 50), rng.uniform(0.5, 1.5, 50))
    with pytest.raises(ValueError):
        vpin(rng.uniform(90, 110, 500) * np.nan, rng.uniform(0.5, 1.5, 500))
    with pytest.raises(ValueError):
        vpin(rng.uniform(90, 110, 500), rng.uniform(0.5, 1.5, 400))
    with pytest.raises(ValueError):
        vpin(np.ones(500), rng.uniform(0.5, 1.5, 500))


def test_bench() -> None:
    out = bench_vpin()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
