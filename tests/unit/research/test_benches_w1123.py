"""Wave-1123 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1123 import (
    bench_contact_linguistics_family,
    bench_descriptive_linguistics_family,
    bench_dialectometry_family,
    bench_etymology_family,
    bench_lexicography_family,
    bench_philological_studies_family,
)

_FAMILY_BENCHES = [
    bench_contact_linguistics_family,
    bench_descriptive_linguistics_family,
    bench_philological_studies_family,
    bench_etymology_family,
    bench_dialectometry_family,
    bench_lexicography_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
