"""Wave-1102 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1102 import (
    bench_geophysics_applied_family,
    bench_hydrogeology_family,
    bench_mineralogy_family,
    bench_sedimentology_family,
    bench_tectonics_family,
    bench_volcanology_family,
)

_FAMILY_BENCHES = [
    bench_mineralogy_family,
    bench_volcanology_family,
    bench_sedimentology_family,
    bench_tectonics_family,
    bench_hydrogeology_family,
    bench_geophysics_applied_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
