"""Wave-1178 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1178 import (
    bench_cognitive_science_2_family,
    bench_complexity_science_family,
    bench_futures_studies_family,
    bench_human_computer_interaction_family,
    bench_interdisciplinary_studies_family,
    bench_systems_science_family,
)

_FAMILY_BENCHES = [
    bench_interdisciplinary_studies_family,
    bench_cognitive_science_2_family,
    bench_futures_studies_family,
    bench_complexity_science_family,
    bench_systems_science_family,
    bench_human_computer_interaction_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
