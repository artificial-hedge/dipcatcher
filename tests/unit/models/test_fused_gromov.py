import numpy as np

from quant_fund.models.fused_gromov import bench_fused_gromov, fused_gromov


def test_fgw_shapes():
    rng = np.random.default_rng(0)
    n = 6
    f = rng.random((n, 2))
    c = np.sqrt(((f[:, None] - f[None, :]) ** 2).sum(-1))
    p = np.full(n, 1 / n)
    t, fgw = fused_gromov(f, f, c, c, p, p, alpha=0.5, eps=0.01, max_iter=20)
    assert t.shape == (n, n)
    assert fgw >= 0.0


def test_bench_sep():
    out = bench_fused_gromov(seed=7)
    assert out["synthetic_fgw_shuffled"] >= out["synthetic_fgw_aligned"]
