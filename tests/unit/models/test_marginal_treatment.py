import numpy as np
import pytest

from quant_fund.models.marginal_treatment import (
    bench_marginal_treatment,
    marginal_te,
    synth_mte,
)


def test_mte_recovers_ate() -> None:
    d = synth_mte(seed=0)
    out = marginal_te(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["x"]),
    )
    assert abs(out["ate_mte"] - 0.5) < 0.2


def test_naive_selection_biased() -> None:
    d = synth_mte(seed=1)
    out = marginal_te(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["x"]),
    )
    assert out["naive_diff"] < 0


def test_mte_range_varies() -> None:
    d = synth_mte(seed=2)
    out = marginal_te(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["x"]),
    )
    assert out["mte_max"] > out["mte_min"]


def test_zero_effect() -> None:
    d = synth_mte(beta=0.0, seed=3)
    out = marginal_te(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["z"]),
        np.asarray(d["x"]),
    )
    assert abs(out["ate_mte"]) < 0.2


def test_synth_shapes() -> None:
    d = synth_mte(n=500, seed=4)
    assert np.asarray(d["y"]).shape == (500,)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 300
    tt = rng.integers(0, 2, n).astype(float)
    with pytest.raises(ValueError):
        marginal_te(
            rng.normal(0, 1, 50), rng.integers(0, 2, 50).astype(float), rng.normal(0, 1, 50)
        )
    with pytest.raises(ValueError):
        marginal_te(rng.normal(0, 1, n) * np.nan, tt, rng.normal(0, 1, n))
    with pytest.raises(ValueError):
        marginal_te(rng.normal(0, 1, n), rng.integers(0, 3, n).astype(float), rng.normal(0, 1, n))
    with pytest.raises(ValueError):
        marginal_te(rng.normal(0, 1, n), tt, rng.normal(0, 1, 60))


def test_bench() -> None:
    out = bench_marginal_treatment()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
