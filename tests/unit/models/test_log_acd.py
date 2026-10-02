import numpy as np
import pytest

from quant_fund.models.log_acd import bench_log_acd, log_acd_fit, synth_log_acd


def test_bench_log_acd_passes():
    r = bench_log_acd()
    assert r["score"] == 1.0


def test_persistence_recovered():
    x, _ = synth_log_acd(seed=5)
    r = log_acd_fit(x)
    assert r["b"] > 0.4


def test_residuals_whitened():
    x, _ = synth_log_acd(seed=7)
    r = log_acd_fit(x)
    assert abs(r["resid_acf1"]) < 0.3


def test_lognormal_variant_runs():
    x, _ = synth_log_acd(seed=2)
    r = log_acd_fit(x, dist="lognormal")
    assert np.isfinite(r["w"])


def test_rejects_nonpositive():
    x, _ = synth_log_acd(seed=1)
    x[10] = -1.0
    with pytest.raises(ValueError):
        log_acd_fit(x)


def test_rejects_short():
    with pytest.raises(ValueError):
        log_acd_fit(np.ones(50))
