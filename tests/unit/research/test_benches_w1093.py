"""Wave-1093 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1093 import (
    bench_canon_law_family,
    bench_civil_law_family,
    bench_common_law_family,
    bench_maritime_law_family,
    bench_procedural_law_family,
    bench_property_law_family,
)

_FAMILY_BENCHES = [
    bench_civil_law_family,
    bench_common_law_family,
    bench_canon_law_family,
    bench_maritime_law_family,
    bench_property_law_family,
    bench_procedural_law_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
