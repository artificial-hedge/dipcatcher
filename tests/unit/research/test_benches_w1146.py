"""Wave-1146 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1146 import (
    bench_entomology_2_family,
    bench_limnology_family,
    bench_mycology_family,
    bench_parasitology_family,
    bench_virology_family,
    bench_wildlife_biology_family,
)

_FAMILY_BENCHES = [
    bench_virology_family,
    bench_parasitology_family,
    bench_mycology_family,
    bench_entomology_2_family,
    bench_limnology_family,
    bench_wildlife_biology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
