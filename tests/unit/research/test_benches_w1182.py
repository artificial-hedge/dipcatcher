"""Wave-1182 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1182 import (
    bench_ballet_studies_family,
    bench_choreography_2_family,
    bench_dance_pedagogy_family,
    bench_dance_science_family,
    bench_movement_studies_family,
    bench_somatic_practices_family,
)

_FAMILY_BENCHES = [
    bench_ballet_studies_family,
    bench_choreography_2_family,
    bench_dance_pedagogy_family,
    bench_somatic_practices_family,
    bench_dance_science_family,
    bench_movement_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
