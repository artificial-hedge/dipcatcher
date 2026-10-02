"""Adapter test: wave-111 computational-geometry canon benches."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w111 import (
    bench_convex_hull_family,
    bench_delaunay_family,
    bench_frechet_family,
    bench_hausdorff_family,
    bench_icp_family,
    bench_kabsch_family,
)

_ALL = [
    bench_kabsch_family,
    bench_icp_family,
    bench_frechet_family,
    bench_hausdorff_family,
    bench_convex_hull_family,
    bench_delaunay_family,
]

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


@pytest.mark.parametrize("fn", _ALL, ids=[f.__name__ for f in _ALL])
def test_family_finite_forbidden_free(fn):
    out = fn()
    assert out, f"{fn.__name__} empty"
    for k, v in out.items():
        assert not any(b in k.lower() for b in _FORBIDDEN), k
        assert np.isfinite(v), (k, v)


@pytest.mark.parametrize("fn", _ALL, ids=[f.__name__ for f in _ALL])
def test_family_synthetic_labels(fn):
    out = fn()
    assert all(k.startswith("synthetic_") for k in out)
