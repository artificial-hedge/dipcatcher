import numpy as np
import pytest

from quant_fund.models.proximal_causal import (
    bench_proximal_causal,
    proximal_ate,
    synth_proximal,
)


def test_proximal_recovers_effect() -> None:
    d = synth_proximal(seed=0)
    out = proximal_ate(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["w"]),
        np.asarray(d["x"]),
    )
    assert abs(out["ate_proximal"] - 1.0) < 0.2


def test_naive_confounded() -> None:
    d = synth_proximal(seed=1)
    out = proximal_ate(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["w"]),
        np.asarray(d["x"]),
    )
    assert out["naive_diff"] > 1.3


def test_zero_effect() -> None:
    d = synth_proximal(effect=0.0, seed=2)
    out = proximal_ate(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["w"]),
        np.asarray(d["x"]),
    )
    assert abs(out["ate_proximal"]) < 0.25


def test_first_stage_relevance() -> None:
    d = synth_proximal(seed=3)
    out = proximal_ate(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["w"]),
        np.asarray(d["x"]),
    )
    assert out["first_stage_r2"] > 0.2


def test_synth_shapes() -> None:
    d = synth_proximal(n=200, seed=4)
    assert np.asarray(d["z"]).shape == (200,)
    assert np.asarray(d["w"]).shape == (200,)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 200
    tt = rng.integers(0, 2, n).astype(float)
    with pytest.raises(ValueError):
        proximal_ate(
            rng.normal(0, 1, 30),
            rng.integers(0, 2, 30).astype(float),
            rng.normal(0, 1, 30),
            rng.normal(0, 1, 30),
        )
    with pytest.raises(ValueError):
        proximal_ate(
            rng.normal(0, 1, n),
            rng.integers(0, 3, n).astype(float),
            rng.normal(0, 1, n),
            rng.normal(0, 1, n),
        )
    with pytest.raises(ValueError):
        proximal_ate(rng.normal(0, 1, n) * np.nan, tt, rng.normal(0, 1, n), rng.normal(0, 1, n))
    with pytest.raises(ValueError):
        proximal_ate(rng.normal(0, 1, n), tt, rng.normal(0, 1, 50), rng.normal(0, 1, 50))


def test_bench() -> None:
    out = bench_proximal_causal()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
