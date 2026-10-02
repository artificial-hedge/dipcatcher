import numpy as np

from quant_fund.models.interpolative_decomp import (
    bench_interpolative_decomp,
    interpolative_decomp,
    pivoted_qr,
)


def test_pivoted_qr_permutation():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((40, 30))
    q, r, perm = pivoted_qr(a, 10)
    assert sorted(perm) == list(range(30))
    assert np.allclose(q.T @ q, np.eye(40), atol=1e-8)


def test_id_identity_block():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((50, 40))
    b, p, skel = interpolative_decomp(a, 8)
    assert np.allclose(p[:, skel], np.eye(8), atol=1e-10)


def test_bench_identity():
    out = bench_interpolative_decomp(seed=7)
    assert out["synthetic_identity_err"] < 1e-9
    assert out["synthetic_id_err"] > out["synthetic_opt_err"]
