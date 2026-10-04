"""Wave-1015 acoustics canon tests."""

from __future__ import annotations

from quant_fund.models.acoustic_wave_eq import bench_acoustic_wave_eq
from quant_fund.models.doppler_effect import bench_doppler_effect
from quant_fund.models.helmholtz_eq import bench_helmholtz_eq
from quant_fund.models.rayleigh_scattering import bench_rayleigh_scattering
from quant_fund.models.room_acoustics import bench_room_acoustics
from quant_fund.models.sound_absorption import bench_sound_absorption


def test_acoustic_wave_eq():
    assert bench_acoustic_wave_eq()["synthetic_acoustic_wave_eq"] == 1.0


def test_helmholtz_eq():
    assert bench_helmholtz_eq()["synthetic_helmholtz_eq"] == 1.0


def test_sound_absorption():
    assert bench_sound_absorption()["synthetic_sound_absorption"] == 1.0


def test_room_acoustics():
    assert bench_room_acoustics()["synthetic_room_acoustics"] == 1.0


def test_rayleigh_scattering():
    assert bench_rayleigh_scattering()["synthetic_rayleigh_scattering"] == 1.0


def test_doppler_effect():
    assert bench_doppler_effect()["synthetic_doppler_effect"] == 1.0
