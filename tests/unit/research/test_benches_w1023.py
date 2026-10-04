"""Wave-1023 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1023 import (
    bench_arbitrage_pricing_family,
    bench_black_scholes_family,
    bench_capm_model_family,
    bench_corporate_finance_family,
    bench_default_risk_family,
    bench_yield_curve_family,
)

_FAMILY_BENCHES = [
    bench_capm_model_family,
    bench_arbitrage_pricing_family,
    bench_black_scholes_family,
    bench_yield_curve_family,
    bench_default_risk_family,
    bench_corporate_finance_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
