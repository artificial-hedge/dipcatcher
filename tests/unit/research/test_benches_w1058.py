"""Wave-1058 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1058 import (
    bench_aesthetics_family,
    bench_epistemology_family,
    bench_ethics_philosophy_family,
    bench_logic_philosophy_family,
    bench_metaphysics_family,
    bench_philosophy_of_science_family,
)

_FAMILY_BENCHES = [
    bench_metaphysics_family,
    bench_epistemology_family,
    bench_ethics_philosophy_family,
    bench_logic_philosophy_family,
    bench_philosophy_of_science_family,
    bench_aesthetics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
