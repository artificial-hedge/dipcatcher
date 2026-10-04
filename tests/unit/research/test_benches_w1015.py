"""Wave-1015 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1015 import (
    bench_acoustic_wave_eq_family,
    bench_doppler_effect_family,
    bench_helmholtz_eq_family,
    bench_rayleigh_scattering_family,
    bench_room_acoustics_family,
    bench_sound_absorption_family,
)

_FAMILY_BENCHES = [
    bench_acoustic_wave_eq_family,
    bench_helmholtz_eq_family,
    bench_sound_absorption_family,
    bench_room_acoustics_family,
    bench_rayleigh_scattering_family,
    bench_doppler_effect_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
