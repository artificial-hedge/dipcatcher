import numpy as np

from quant_fund.models.qmc_sequences import (
    bench_qmc,
    faure,
    halton,
    hammersley,
    l2_discrepancy,
    mc_uniform,
)


def test_halton_bounds_marginals():
    p = halton(500, 3)
    assert np.all(p >= 0) and np.all(p < 1)
    for j in range(3):
        assert abs(p[:, j].mean() - 0.5) < 0.05


def test_halton_beats_mc_discrepancy():
    n = 625
    assert l2_discrepancy(halton(n, 4)) < l2_discrepancy(mc_uniform(n, 4, seed=0))


def test_faure_beats_mc_discrepancy():
    n = 625
    assert l2_discrepancy(faure(n, 4)) < l2_discrepancy(mc_uniform(n, 4, seed=0))


def test_hammersley_shape():
    p = hammersley(100, 4)
    assert p.shape == (100, 4)
    assert np.allclose(p[:, 0], np.arange(100) / 100)


def test_bench_keys():
    out = bench_qmc()
    assert out["synthetic_disc_halton"] < out["synthetic_disc_mc"]
    assert out["synthetic_disc_faure"] < out["synthetic_disc_mc"]
