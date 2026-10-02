"""Tests for wave-115 UQ canon: smolyak, pce, bayesian_quadrature,
kl_expand, active_subspace, mimc."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.active_subspace import (
    active_subspace,
    bench_active_subspace,
)
from quant_fund.models.bayesian_quadrature import (
    bayesian_quadrature,
    bench_bayesian_quadrature,
)
from quant_fund.models.kl_expand import (
    bench_kl_expand,
    kl_decompose,
    kl_energy_order,
    kl_reconstruct,
)
from quant_fund.models.mimc import bench_mimc, mimc_estimate
from quant_fund.models.pce import bench_pce, pce_fit, pce_sobol
from quant_fund.models.smolyak import bench_smolyak, smolyak_integrate


def test_smolyak_constant():
    val, _ = smolyak_integrate(lambda p: 3.7, 2, 2)
    assert val == pytest.approx(3.7)


def test_smolyak_better_with_level():
    f = lambda p: float(np.exp(p[0] + p[1]))  # noqa: E731
    e1, _ = smolyak_integrate(f, 2, 1)
    e4, _ = smolyak_integrate(f, 2, 4)
    exact = (np.e - 1) ** 2
    assert abs(e4 - exact) < abs(e1 - exact)


def test_pce_quadratic():
    c, idx = pce_fit(lambda p: float(p[0] ** 2), 1, 3)
    # E[x²]=1/3, Var(x²)=4/45
    assert abs(c[0] - 1 / 3) < 1e-6
    assert abs(np.sum(c[1:] ** 2) - 4 / 45) < 1e-6


def test_pce_sobol_additive():
    c, idx = pce_fit(lambda p: float(p[0] + 2 * p[1] ** 2), 2, 4)
    s = pce_sobol(c, idx, 2)
    assert s.sum() == pytest.approx(1.0, abs=1e-6)


def test_bq_linear():
    mean, _, _ = bayesian_quadrature(lambda x: float(x[0]), 1, 4, ls=0.5)
    assert mean == pytest.approx(0.5, abs=0.05)


def test_kl_decompose():
    t = np.linspace(0, 1, 50)
    cov = np.exp(-np.abs(t[:, None] - t[None, :]) / 0.3)
    w, v = kl_decompose(cov)
    assert np.all(np.diff(w) <= 1e-9)
    assert kl_energy_order(w, 0.9) < 50
    rng = np.random.default_rng(0)
    path = v[:, :5] @ rng.normal(size=5)
    rec = kl_reconstruct(path, v, 5)
    assert np.linalg.norm(path - rec) < 1e-10


def test_active_subspace_linear():
    rng = np.random.default_rng(1)
    w, v, gaps = active_subspace(lambda x: float(x[0] * 2 + x[1]), 2, 50, rng)
    assert abs(v[0, 0]) > 0.8  # dominant direction ≈ x1


def test_mimc_telescope():
    rng = np.random.default_rng(2)
    fn = lambda a, b, r: 1.0 + 0.1 * (a + b)  # noqa: E731
    est, _ = mimc_estimate(fn, [(0, 0), (1, 0), (0, 1), (1, 1)], 10, rng)
    # Δ(1,1)=P11−P10−P01+P00; total=P00+Δ10+Δ01+Δ11 = P11
    assert est == pytest.approx(fn(1, 1, rng), abs=1e-6)


@pytest.mark.parametrize(
    "fn",
    [
        bench_smolyak,
        bench_pce,
        bench_bayesian_quadrature,
        bench_kl_expand,
        bench_active_subspace,
        bench_mimc,
    ],
    ids=lambda f: f.__name__,
)
def test_w115_benches(fn):
    out = fn(seed=20261231)
    assert out and all(np.isfinite(v) for v in out.values())
    assert all(k.startswith("synthetic_") for k in out)
