"""Wave-1048 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1048 import (
    bench_animal_surgery_family,
    bench_equine_medicine_family,
    bench_veterinary_anatomy_family,
    bench_veterinary_epidemiology_family,
    bench_veterinary_pathology_family,
    bench_veterinary_pharmacology_family,
)

_FAMILY_BENCHES = [
    bench_veterinary_anatomy_family,
    bench_veterinary_pathology_family,
    bench_veterinary_pharmacology_family,
    bench_animal_surgery_family,
    bench_veterinary_epidemiology_family,
    bench_equine_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
