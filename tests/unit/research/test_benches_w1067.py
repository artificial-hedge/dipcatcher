"""Wave-1067 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1067 import (
    bench_archival_studies_family,
    bench_digital_humanities_family,
    bench_information_science_family,
    bench_knowledge_organization_family,
    bench_library_science_family,
    bench_museum_studies_family,
)

_FAMILY_BENCHES = [
    bench_library_science_family,
    bench_information_science_family,
    bench_archival_studies_family,
    bench_museum_studies_family,
    bench_digital_humanities_family,
    bench_knowledge_organization_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
