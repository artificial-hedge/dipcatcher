"""Tests for clt_classic — CLT bench incl. heavy-tail counterexample."""

from __future__ import annotations

import numpy as np

from quant_fund.models.clt_classic import bench_clt_classic, clt_z


def test_cauchy_breaks_clt_scaling():
    # the bench must actually exercise the infinite-variance case —
    # a literal True marker verified nothing.
    from quant_fund.models.clt_classic import _clt_fails_cauchy

    assert _clt_fails_cauchy(np.random.default_rng(0))


def test_finite_variance_concentrates():
    rng = np.random.default_rng(1)
    z = [clt_z(4000, rng) for _ in range(80)]
    assert 0.3 < float(np.var(z)) < 3.0


def test_bench():
    assert bench_clt_classic()["synthetic_clt_classic"] == 1.0
