"""Wave-1208 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1208 import (
    bench_child_adolescent_therapy_family,
    bench_couples_therapy_family,
    bench_family_therapy_family,
    bench_group_therapy_family,
    bench_marriage_family_therapy_family,
    bench_trauma_therapy_family,
)

_FAMILY_BENCHES = [
    bench_marriage_family_therapy_family,
    bench_group_therapy_family,
    bench_couples_therapy_family,
    bench_family_therapy_family,
    bench_child_adolescent_therapy_family,
    bench_trauma_therapy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
