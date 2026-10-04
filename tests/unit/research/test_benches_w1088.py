"""Wave-1088 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1088 import (
    bench_analytic_philosophy_family,
    bench_ancient_philosophy_family,
    bench_continental_philosophy_family,
    bench_existentialism_family,
    bench_medieval_philosophy_family,
    bench_pragmatism_family,
)

_FAMILY_BENCHES = [
    bench_ancient_philosophy_family,
    bench_medieval_philosophy_family,
    bench_continental_philosophy_family,
    bench_analytic_philosophy_family,
    bench_pragmatism_family,
    bench_existentialism_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
