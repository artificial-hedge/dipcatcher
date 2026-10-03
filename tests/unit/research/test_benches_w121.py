"""Adapter test: wave-121 best-arm-ID canon benches."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w121 import (
    bench_lil_ucb_family,
    bench_median_elim_family,
    bench_sequential_halving_family,
    bench_track_stop_family,
    bench_ttts_family,
    bench_ugape_family,
)

_ALL = [
    bench_lil_ucb_family,
    bench_sequential_halving_family,
    bench_median_elim_family,
    bench_ugape_family,
    bench_ttts_family,
    bench_track_stop_family,
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
