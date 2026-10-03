"""Wave-313 numerical-4/multigrid module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.amg_lite import interp_matrix, poisson_2d, select_coarse, strength_matrix
from quant_fund.models.bicgstab import bicgstab
from quant_fund.models.chebyshev_iter import chebyshev_3term
from quant_fund.models.ilu_precond import ilu0, solve_lu
from quant_fund.models.minres import minres
from quant_fund.models.v_cycle import poisson_1d, prolong_linear, restrict_full


def test_restrict_prolong_shapes() -> None:
    r = np.random.default_rng(0).normal(size=127)
    rc = restrict_full(r)
    assert len(rc) == 63
    e = prolong_linear(rc, 127)
    assert e.shape == (127,) and np.isfinite(e).all()


def test_amg_split_rows() -> None:
    a = poisson_2d(6)
    s = strength_matrix(a)
    is_c = select_coarse(s)
    p = interp_matrix(a, s, is_c)
    assert p.shape[1] == int(is_c.sum())
    assert np.allclose(p.sum(axis=1), 1.0)  # partition of unity


def test_bicgstab_residual() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(20, 20)) * 0.3 + np.diag(np.full(20, 4.0))
    b = rng.normal(size=20)
    x, hist = bicgstab(a, b, np.zeros(20))
    assert hist[-1] < 1e-10 * hist[0]
    assert np.linalg.norm(x - np.linalg.solve(a, b)) < 1e-8


def test_minres_monotone() -> None:
    rng = np.random.default_rng(2)
    q_, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    e = np.concatenate([np.linspace(1, 4, 12), -np.linspace(1, 4, 8)])
    a = q_ @ np.diag(e) @ q_.T
    b = rng.normal(size=20)
    _, hist = minres(a, b, 20)
    assert np.all(np.diff(np.asarray(hist)) <= 1e-10)


def test_ilu0_pattern_and_solve() -> None:
    a = poisson_1d(10)
    lo, u = ilu0(a)
    assert np.allclose(np.tril(lo), lo) and np.allclose(np.triu(u), u)
    y = np.random.default_rng(3).normal(size=10)
    lu = lo @ u
    assert np.allclose(lu @ solve_lu(lo, u, y), lu @ np.linalg.solve(lu, y))


def test_chebyshev_rate() -> None:
    rng = np.random.default_rng(4)
    q_, _ = np.linalg.qr(rng.normal(size=(30, 30)))
    eigs = rng.uniform(1.0, 50.0, 30)
    eigs[0], eigs[-1] = 1.0, 50.0
    a = q_ @ np.diag(eigs) @ q_.T
    b = rng.normal(size=30)
    _, hist = chebyshev_3term(a, b, 1.0, 50.0, 60)
    assert hist[-1] < 1e-6 * hist[0]
