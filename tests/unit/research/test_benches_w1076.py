"""Wave-1076 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1076 import (
    bench_archaeometry_family,
    bench_bioarchaeology_family,
    bench_experimental_archaeology_family,
    bench_field_archaeology_family,
    bench_landscape_archaeology_family,
    bench_underwater_archaeology_family,
)

_FAMILY_BENCHES = [
    bench_field_archaeology_family,
    bench_archaeometry_family,
    bench_bioarchaeology_family,
    bench_underwater_archaeology_family,
    bench_landscape_archaeology_family,
    bench_experimental_archaeology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
