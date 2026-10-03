"""Tests for the wave-31 scorecard adapters (research/benches_w31.py)."""

from __future__ import annotations

import math

import pytest

from quant_fund.research import benches_w31
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FORBIDDEN = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})

ADAPTERS = [
    benches_w31.bench_kernel_iv,
    benches_w31.bench_multistate,
    benches_w31.bench_partial_linear,
    benches_w31.bench_heckman,
    benches_w31.bench_rd,
    benches_w31.bench_bounds,
]

FAMILIES = {"kernel_iv", "multistate", "partial_linear", "heckman", "rd", "bounds"}


def test_families_registered():
    assert FAMILIES <= OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("adapter", ADAPTERS, ids=[f.__name__ for f in ADAPTERS])
def test_adapter_emits_finite_synthetic(adapter):
    out = adapter()
    assert isinstance(out, dict)
    assert out, f"{adapter.__name__} degraded to {{}}"
    for k, v in out.items():
        assert isinstance(k, str)
        assert _FORBIDDEN.isdisjoint(k.lower().split("_")), k
        assert math.isfinite(v), k


def test_kernel_iv_recovers():
    out = benches_w31.bench_kernel_iv()
    assert out["synthetic_iv_relerr"] < out["synthetic_naive_relerr"]


def test_multistate_intensity():
    out = benches_w31.bench_multistate()
    assert out["synthetic_intensity_relerr"] < 0.3
    assert out["synthetic_pmat_relerr"] < 0.15


def test_partial_linear_beta():
    out = benches_w31.bench_partial_linear()
    assert out["synthetic_beta_err"] < 0.1
    assert out["synthetic_g_corr"] > 0.9


def test_heckman_twostep():
    out = benches_w31.bench_heckman()
    assert out["synthetic_beta_twostep_err"] < out["synthetic_beta_ols_err"]


def test_rd_recovers_tau():
    out = benches_w31.bench_rd()
    assert out["synthetic_tau_err"] < 0.5
    assert out["synthetic_tau_t"] > 3.0
    assert out["synthetic_mccrary_flags"] == 1.0


def test_bounds_cover_truth():
    out = benches_w31.bench_bounds()
    assert out["synthetic_manski_covers"] == 1.0
    assert out["synthetic_lee_covers"] == 1.0


def test_determinism_each():
    for adapter in ADAPTERS:
        assert adapter() == adapter()
