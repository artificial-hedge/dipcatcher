"""Wave-1172 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1172 import (
    bench_accounting_3_family,
    bench_entrepreneurship_3_family,
    bench_finance_5_family,
    bench_management_3_family,
    bench_marketing_3_family,
    bench_organizational_behavior_family,
)

_FAMILY_BENCHES = [
    bench_management_3_family,
    bench_marketing_3_family,
    bench_accounting_3_family,
    bench_finance_5_family,
    bench_entrepreneurship_3_family,
    bench_organizational_behavior_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
