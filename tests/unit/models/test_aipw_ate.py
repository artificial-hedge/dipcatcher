import numpy as np
import pytest

from quant_fund.models.aipw_ate import aipw_ate, bench_aipw_ate, synth_observational


def test_aipw_recovers_ate() -> None:
    d = synth_observational(ate=0.7, seed=0)
    out = aipw_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    assert abs(out["ate"] - 0.7) < 0.15


def test_beats_naive() -> None:
    d = synth_observational(ate=0.7, seed=1)
    out = aipw_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    y, dd = np.asarray(d["y"]), np.asarray(d["treat"])
    naive = float(y[dd == 1].mean() - y[dd == 0].mean())
    assert abs(out["ate"] - 0.7) < abs(naive - 0.7)


def test_true_propensity_also_unbiased() -> None:
    d = synth_observational(ate=0.7, seed=2)
    out = aipw_ate(
        np.asarray(d["y"]),
        np.asarray(d["treat"]),
        np.asarray(d["x"]),
        propensity=np.asarray(d["e_true"]),
    )
    assert abs(out["ate"] - 0.7) < 0.15


def test_se_positive_and_overlap() -> None:
    d = synth_observational(seed=3)
    out = aipw_ate(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["x"]))
    assert out["se"] > 0
    assert out["e_min"] >= 0.01 - 1e-9 and out["e_max"] <= 0.99 + 1e-9


def test_synth_shapes() -> None:
    d = synth_observational(n=500, seed=4)
    assert np.asarray(d["y"]).shape == (500,)
    assert np.asarray(d["x"]).shape == (500, 3)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    y = rng.normal(0, 1, n)
    tr = rng.binomial(1, 0.5, n).astype(float)
    with pytest.raises(ValueError):
        aipw_ate(y[:50], tr[:50], x[:50])
    with pytest.raises(ValueError):
        aipw_ate(y * np.nan, tr, x)
    with pytest.raises(ValueError):
        aipw_ate(y, np.full(n, 1.0), x)
    with pytest.raises(ValueError):
        aipw_ate(y, tr, x, propensity=rng.normal(0, 1, 50))
    with pytest.raises(ValueError):
        aipw_ate(y, tr, x, trim=0.5)


def test_bench() -> None:
    out = bench_aipw_ate()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
