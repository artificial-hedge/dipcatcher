"""Chain-ladder / Mack / BF / bootstrap reserving tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.chain_ladder import (
    _synthetic_triangle,
    ata_factors,
    bench_chain_ladder,
    bornhuetter_ferguson,
    mack_variance,
    odp_bootstrap,
    reserves,
    ultimate,
)


def test_factors_recover_known_pattern():
    tri = _synthetic_triangle(11)
    f = ata_factors(tri)
    assert f.shape == (9,)
    assert np.all(f > 0.9)
    assert f[0] > f[5]  # early dev dominates late dev
    assert f[-2:].mean() < 1.15


def test_ultimate_exceeds_diagonal():
    tri = _synthetic_triangle(12)
    ult = ultimate(tri)
    diag = np.diag(np.fliplr(tri))
    assert (ult >= diag - 1e-9).all()


def test_reserve_positive():
    tri = _synthetic_triangle(13)
    r = reserves(tri)
    assert r["reserve_total"] > 0


def test_mack_se_positive_and_bounded():
    tri = _synthetic_triangle(14)
    mk = mack_variance(tri)
    assert mk["se_reserve"] > 0
    assert 0 < mk["cv_reserve"] < 0.5


def test_bf_reasonable():
    tri = _synthetic_triangle(15)
    prem = np.full(10, 16000.0)
    bf = bornhuetter_ferguson(tri, prem, elr=0.9)
    assert bf["bf_reserve_total"] > 0
    assert 0.5 < bf["bf_vs_cl_ratio"] < 2.0


def test_bootstrap_se_positive():
    tri = _synthetic_triangle(16)
    bt = odp_bootstrap(tri, n_boot=30, seed=3)
    assert bt["boot_reserve_se"] > 0
    assert bt["boot_reserve_mean"] > 0


def test_fail_closed():
    with pytest.raises(ValueError):
        ata_factors(np.ones((4, 5)))
    with pytest.raises(ValueError):
        ultimate(_synthetic_triangle(1), np.ones(3))
    with pytest.raises(ValueError):
        bornhuetter_ferguson(_synthetic_triangle(2), np.zeros(10))
    bad = _synthetic_triangle(3)
    bad[0, -1] = np.nan  # corrupt anti-diagonal
    with pytest.raises(ValueError):
        reserves(bad)


def test_bench_passes():
    out = bench_chain_ladder()
    assert out["synthetic_reserve_total"] > 0
    assert out["synthetic_mack_cv"] > 0


def test_mack_includes_cross_year_covariance():
    """Aggregate MSE must exceed the sum of per-year proc+param terms —
    reserves share fitted factors so their estimation errors correlate
    (the docstring claims Mack's covariance cross-terms; they were
    silently missing from the total)."""
    from quant_fund.models.chain_ladder import _check_triangle

    tri = _synthetic_triangle(17)
    ca = _check_triangle(tri)
    n = ca.shape[0]
    f = ata_factors(ca)
    sig2 = np.empty(n - 1)
    for j in range(n - 1):
        col, nxt = ca[:, j], ca[:, j + 1]
        m = np.isfinite(col) & np.isfinite(nxt) & (col > 0)
        w = col[m]
        dev = nxt[m] / col[m] - f[j]
        if m.sum() < 2:
            sig2[j] = min(sig2[:j]) if j else 1.0
        else:
            sig2[j] = float((w * dev * dev).sum() / (m.sum() - 1))
    s_cols = np.array(
        [
            sum(ca[k, j] for k in range(n - j - 1) if np.isfinite(ca[k, j]) and ca[k, j] > 0)
            for j in range(n - 1)
        ]
    )
    diag = np.diag(np.fliplr(ca))
    no_cross = 0.0
    for i in range(1, n):
        c_k = diag[i]
        for j in range(n - 1 - i, n - 1):
            no_cross += c_k * sig2[j] / f[j] + c_k * c_k * (sig2[j] / (f[j] * f[j])) / s_cols[j]
            c_k = c_k * f[j]
    mk = mack_variance(tri)
    assert mk["mse_total"] > no_cross * 1.001
