"""Wave-1112 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1112 import (
    bench_biophysics_family,
    bench_comparative_anatomy_family,
    bench_developmental_biology_family,
    bench_ethology_family,
    bench_evolutionary_biology_family,
    bench_neurobiology_family,
)

_FAMILY_BENCHES = [
    bench_biophysics_family,
    bench_evolutionary_biology_family,
    bench_developmental_biology_family,
    bench_neurobiology_family,
    bench_ethology_family,
    bench_comparative_anatomy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
