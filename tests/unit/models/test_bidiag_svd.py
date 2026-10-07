import numpy as np

from quant_fund.models.bidiag_svd import _bidiag, bench_bidiag_svd


def _offmask(B: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(B, dtype=bool)
    for i in range(min(B.shape)):
        mask[i, i] = True
        if i + 1 < B.shape[1]:
            mask[i, i + 1] = True
    return B[~mask]


def test_bidiag_zero_pivot_column():
    # x[0] == 0 must not collapse the Householder reflector (np.sign(0)=0):
    # column below the diagonal has to be zeroed anyway.
    B = _bidiag(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert abs(B[1, 0]) < 1e-12
    np.testing.assert_allclose(
        np.linalg.svd(B, compute_uv=False),
        np.linalg.svd(np.array([[0.0, 1.0], [1.0, 0.0]]), compute_uv=False),
    )


def test_bidiag_zero_pivot_row():
    # same edge case on the row reflector: first element of the row is 0.
    a = np.eye(4)
    a[0, 3] = 1.0
    a[1, 0] = 0.5
    a[2, 1] = 0.3
    B = _bidiag(a.copy())
    assert np.linalg.norm(_offmask(B)) < 1e-9


def test_bidiag_random_square():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 6))
    B = _bidiag(a.copy())
    assert np.linalg.norm(_offmask(B)) < 1e-9
    np.testing.assert_allclose(
        np.linalg.svd(B, compute_uv=False), np.linalg.svd(a, compute_uv=False), atol=1e-8
    )


def test_bench_contract():
    out = bench_bidiag_svd()
    assert out["synthetic_bidiag_sv_err"] < 1e-8
    assert out["synthetic_bidiag_offmass"] < 1e-8
