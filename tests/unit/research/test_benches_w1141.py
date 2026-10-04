"""Wave-1141 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1141 import (
    bench_astrobiology_family,
    bench_astrochemistry_family,
    bench_cosmology_2_family,
    bench_exoplanet_science_family,
    bench_galactic_dynamics_family,
    bench_helio_seismology_family,
)

_FAMILY_BENCHES = [
    bench_cosmology_2_family,
    bench_astrobiology_family,
    bench_astrochemistry_family,
    bench_helio_seismology_family,
    bench_exoplanet_science_family,
    bench_galactic_dynamics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
