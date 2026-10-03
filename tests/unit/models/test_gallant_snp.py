import numpy as np
import pytest

from quant_fund.models.gallant_snp import bench_gallant_snp, snp_density, snp_fit, synth_snp


def test_bench_gallant_snp_passes():
    r = bench_gallant_snp()
    assert r["score"] == 1.0


def test_snp_t_gains_over_gauss():
    t_samp, _ = synth_snp(seed=5)
    r = snp_fit(t_samp, k=4)
    assert r["ll_gain"] > 5.0


def test_snp_normal_small_gain():
    _, n_samp = synth_snp(seed=6)
    r = snp_fit(n_samp, k=4)
    assert abs(r["ll_gain"]) < 15.0


def test_snp_density_nonnegative():
    t_samp, _ = synth_snp(seed=2, n=800)
    grid = np.linspace(-4, 4, 50)
    r = snp_density(t_samp, grid, k=3)
    assert np.all(np.asarray(r["density"]) >= 0.0)


def test_snp_rejects_short():
    with pytest.raises(ValueError):
        snp_fit(np.arange(50.0))
