import numpy as np
import pytest

from quant_fund.metrics.kiefer_vogelsang import (
    bench_kiefer_vogelsang,
    fixedb_cv,
    fixedb_pvalue,
    fixedb_t,
    synth_ar1_mean,
)


def test_cv_approaches_normal_small_b() -> None:
    cv = fixedb_cv(0.02, 0.05)
    assert 1.7 < cv < 2.5


def test_cv_increases_with_b() -> None:
    assert fixedb_cv(0.05, 0.05) < fixedb_cv(0.5, 0.05)


def test_shifted_mean_rejects() -> None:
    x = synth_ar1_mean(mean=0.9, seed=0)
    assert fixedb_pvalue(x, b=0.1) < 0.05


def test_null_reasonable() -> None:
    x = synth_ar1_mean(mean=0.0, seed=0)
    assert fixedb_pvalue(x, b=0.1) > 0.05


def test_fixedb_t_keys() -> None:
    x = synth_ar1_mean(seed=3)
    out = fixedb_t(x, b=0.2)
    assert out["j_hat"] > 0
    assert out["m"] == pytest.approx(0.2 * 300, abs=1)


def test_validation() -> None:
    with pytest.raises(ValueError):
        fixedb_t(np.ones(10))
    with pytest.raises(ValueError):
        fixedb_t(np.ones(60), b=2.0)
    with pytest.raises(ValueError):
        fixedb_t(np.ones(60) * np.nan)
    with pytest.raises(ValueError):
        fixedb_cv(0.1, level=0.9)


def test_bench() -> None:
    out = bench_kiefer_vogelsang()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
