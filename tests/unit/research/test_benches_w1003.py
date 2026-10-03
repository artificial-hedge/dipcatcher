"""Wave-1003 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1003 import (
    bench_alfven_waves_family,
    bench_elsaesser_vars_family,
    bench_frozen_flux_family,
    bench_magnetic_reconnection_family,
    bench_mhd_equations_family,
    bench_parker_solar_wind_family,
)

_FAMILY_BENCHES = [
    bench_mhd_equations_family,
    bench_alfven_waves_family,
    bench_parker_solar_wind_family,
    bench_magnetic_reconnection_family,
    bench_frozen_flux_family,
    bench_elsaesser_vars_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
