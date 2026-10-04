"""Wave-1081 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1081 import (
    bench_baroque_studies_family,
    bench_early_modern_family,
    bench_enlightenment_studies_family,
    bench_humanism_family,
    bench_reformation_studies_family,
    bench_renaissance_studies_family,
)

_FAMILY_BENCHES = [
    bench_renaissance_studies_family,
    bench_early_modern_family,
    bench_humanism_family,
    bench_reformation_studies_family,
    bench_baroque_studies_family,
    bench_enlightenment_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
