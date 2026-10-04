"""Wave-1092 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1092 import (
    bench_connoisseurship_family,
    bench_curation_practice_family,
    bench_formal_analysis_family,
    bench_iconography_family,
    bench_iconology_family,
    bench_provenance_studies_family,
)

_FAMILY_BENCHES = [
    bench_iconography_family,
    bench_iconology_family,
    bench_connoisseurship_family,
    bench_provenance_studies_family,
    bench_curation_practice_family,
    bench_formal_analysis_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
