"""Adapter test: wave-110 digital-comms canon benches."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w110 import (
    bench_costas_family,
    bench_gardner_family,
    bench_gf256_family,
    bench_reed_solomon_family,
    bench_rrc_filter_family,
    bench_viterbi_decode_family,
)

_ALL = [
    bench_viterbi_decode_family,
    bench_gf256_family,
    bench_reed_solomon_family,
    bench_costas_family,
    bench_gardner_family,
    bench_rrc_filter_family,
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
