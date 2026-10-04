"""Wave-1063 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1063 import (
    bench_communication_theory_family,
    bench_digital_media_family,
    bench_journalism_family,
    bench_media_studies_family,
    bench_public_relations_family,
    bench_rhetoric_family,
)

_FAMILY_BENCHES = [
    bench_media_studies_family,
    bench_journalism_family,
    bench_public_relations_family,
    bench_rhetoric_family,
    bench_communication_theory_family,
    bench_digital_media_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
