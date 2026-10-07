import numpy as np
import pytest

from quant_fund.models.garch_in_mean import (
    bench_garch_in_mean,
    garchm_fit,
    synth_garchm,
)


def test_bench_garchm_passes():
    r = bench_garch_in_mean()
    assert r["synthetic_score"] == 1.0


def test_lam_positive_on_planted():
    y, _ = synth_garchm(seed=7)
    r = garchm_fit(y)
    assert r["lam"] > 0.0


def test_control_lam_near_zero():
    _, y0 = synth_garchm(seed=9)
    r = garchm_fit(y0)
    assert abs(r["lam"]) < 0.6


def test_variance_persistence_lt1():
    y, _ = synth_garchm(seed=2)
    r = garchm_fit(y)
    assert r["a"] + r["b"] < 1.01


def test_rejects_short():
    with pytest.raises(ValueError):
        garchm_fit(np.zeros(100))


def test_sqrth_variant_runs():
    y, _ = synth_garchm(seed=3)
    r = garchm_fit(y, mean_kind="sqrth")
    assert np.isfinite(r["lam"])
