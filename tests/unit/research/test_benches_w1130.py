"""Wave-1130 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1130 import (
    bench_anthropology_of_religion_family,
    bench_cognitive_anthropology_family,
    bench_kinship_studies_family,
    bench_material_culture_family,
    bench_museum_anthropology_family,
    bench_social_anthropology_family,
)

_FAMILY_BENCHES = [
    bench_social_anthropology_family,
    bench_cognitive_anthropology_family,
    bench_anthropology_of_religion_family,
    bench_kinship_studies_family,
    bench_material_culture_family,
    bench_museum_anthropology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
