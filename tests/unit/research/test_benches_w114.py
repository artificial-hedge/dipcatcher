"""Adapter test: wave-114 GNSS canon benches."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w114 import (
    bench_allan_variance_family,
    bench_gold_code_family,
    bench_klobuchar_family,
    bench_lambda_method_family,
    bench_rtk_family,
    bench_strapdown_family,
)

_ALL = [
    bench_gold_code_family,
    bench_klobuchar_family,
    bench_allan_variance_family,
    bench_strapdown_family,
    bench_lambda_method_family,
    bench_rtk_family,
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
