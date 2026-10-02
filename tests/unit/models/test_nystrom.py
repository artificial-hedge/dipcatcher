import numpy as np

from quant_fund.models.nystrom import bench_nystrom, nystrom


def test_nystrom_psd_out():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 4))
    k_mat = np.exp(-((x[:, None] - x[None, :]) ** 2).sum(-1) / 4)
    idx = rng.choice(60, 15, replace=False)
    approx = nystrom(k_mat, idx)
    assert approx.shape == (60, 60)
    assert np.min(np.linalg.eigvalsh(approx)) > -1e-8


def test_bench_err_bounded():
    out = bench_nystrom(seed=5)
    assert 0.0 <= out["synthetic_nystrom_err"] < 0.5
