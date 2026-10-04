"""Wave-1115 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1115 import (
    bench_financial_economics_family,
    bench_industrial_organization_family,
    bench_international_economics_family,
    bench_labor_economics_family,
    bench_monetary_economics_family,
    bench_public_economics_family,
)

_FAMILY_BENCHES = [
    bench_labor_economics_family,
    bench_public_economics_family,
    bench_industrial_organization_family,
    bench_international_economics_family,
    bench_financial_economics_family,
    bench_monetary_economics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
