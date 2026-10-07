import numpy as np

from quant_fund.models.block_lanczos import bench_block_lanczos, block_lanczos


def test_ritz_values_inside_spectrum():
    # Ritz values of the block tridiagonal T = Q'AQ are bounded by the
    # spectrum of A (interlacing). Off-diagonal blocks must be the B
    # factors, not the identity — with identity blocks the "Ritz" values
    # escape [lambda_min, lambda_max].
    rng = np.random.RandomState(1)
    a = rng.rand(20, 20)
    a = a @ a.T
    tru = np.linalg.eigvalsh(a)
    ritz = block_lanczos(a, 4, 5, rng)
    assert ritz.min() >= tru.min() - 1e-8
    assert ritz.max() <= tru.max() + 1e-8


def test_ritz_approximates_extremes():
    rng = np.random.RandomState(2)
    a = rng.rand(24, 24)
    a = a @ a.T
    tru = np.linalg.eigvalsh(a)
    ritz = block_lanczos(a, 4, 6, rng)
    assert abs(ritz.max() - tru.max()) / tru.max() < 0.05
    assert abs(ritz.min() - tru.min()) / tru.min() < 0.05


def test_exact_when_krylov_covers():
    # identity: a single block step resolves the whole spectrum
    rng = np.random.RandomState(3)
    ritz = block_lanczos(np.eye(6), 6, 1, rng)
    np.testing.assert_allclose(ritz, np.ones(6), atol=1e-10)


def test_bench_contract():
    out = bench_block_lanczos()
    assert out["synthetic_block_lanczos_top"] > 0.9
