"""Wave-1096 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1096 import (
    bench_conservation_biology_family,
    bench_environmental_toxicology_family,
    bench_landscape_ecology_family,
    bench_marine_conservation_family,
    bench_pollution_science_family,
    bench_urban_ecology_family,
)

_FAMILY_BENCHES = [
    bench_pollution_science_family,
    bench_conservation_biology_family,
    bench_environmental_toxicology_family,
    bench_urban_ecology_family,
    bench_landscape_ecology_family,
    bench_marine_conservation_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
