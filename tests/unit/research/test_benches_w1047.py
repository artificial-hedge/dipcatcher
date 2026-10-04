"""Wave-1047 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1047 import (
    bench_aquaculture_family,
    bench_benthic_biology_family,
    bench_coral_reef_ecology_family,
    bench_fisheries_science_family,
    bench_marine_ecology_family,
    bench_plankton_dynamics_family,
)

_FAMILY_BENCHES = [
    bench_plankton_dynamics_family,
    bench_marine_ecology_family,
    bench_fisheries_science_family,
    bench_aquaculture_family,
    bench_benthic_biology_family,
    bench_coral_reef_ecology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
