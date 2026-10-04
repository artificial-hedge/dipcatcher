"""Wave-1019 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1019 import (
    bench_earthquake_magnitude_family,
    bench_geomagnetism_family,
    bench_gravity_anomaly_family,
    bench_heat_flow_geo_family,
    bench_plate_tectonics_family,
    bench_seismic_waves_family,
)

_FAMILY_BENCHES = [
    bench_seismic_waves_family,
    bench_earthquake_magnitude_family,
    bench_plate_tectonics_family,
    bench_gravity_anomaly_family,
    bench_geomagnetism_family,
    bench_heat_flow_geo_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
