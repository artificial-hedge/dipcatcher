"""Adapter test: wave-112 DSP canon benches."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.benches_w112 import (
    bench_biquad_family,
    bench_farrow_family,
    bench_filtfilt_family,
    bench_iir_design_family,
    bench_remez_family,
    bench_resample_poly_family,
)

_ALL = [
    bench_remez_family,
    bench_iir_design_family,
    bench_biquad_family,
    bench_filtfilt_family,
    bench_resample_poly_family,
    bench_farrow_family,
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
