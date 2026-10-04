"""Wave-1106 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1106 import (
    bench_biomaterials_family,
    bench_characterization_methods_family,
    bench_composite_materials_family,
    bench_phase_diagrams_family,
    bench_semiconductors_materials_family,
    bench_thin_films_family,
)

_FAMILY_BENCHES = [
    bench_semiconductors_materials_family,
    bench_composite_materials_family,
    bench_thin_films_family,
    bench_biomaterials_family,
    bench_phase_diagrams_family,
    bench_characterization_methods_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
