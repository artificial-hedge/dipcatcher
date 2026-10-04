"""Wave-1015 acoustics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.acoustic_wave_eq import bench_acoustic_wave_eq
from quant_fund.models.doppler_effect import bench_doppler_effect
from quant_fund.models.helmholtz_eq import bench_helmholtz_eq
from quant_fund.models.rayleigh_scattering import bench_rayleigh_scattering
from quant_fund.models.room_acoustics import bench_room_acoustics
from quant_fund.models.sound_absorption import bench_sound_absorption

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_acoustic_wave_eq_family(seed: int = _SEED + 40300) -> dict[str, float]:
    return _finite_blob(bench_acoustic_wave_eq(seed))


def bench_helmholtz_eq_family(seed: int = _SEED + 40301) -> dict[str, float]:
    return _finite_blob(bench_helmholtz_eq(seed))


def bench_sound_absorption_family(seed: int = _SEED + 40302) -> dict[str, float]:
    return _finite_blob(bench_sound_absorption(seed))


def bench_room_acoustics_family(seed: int = _SEED + 40303) -> dict[str, float]:
    return _finite_blob(bench_room_acoustics(seed))


def bench_rayleigh_scattering_family(seed: int = _SEED + 40304) -> dict[str, float]:
    return _finite_blob(bench_rayleigh_scattering(seed))


def bench_doppler_effect_family(seed: int = _SEED + 40305) -> dict[str, float]:
    return _finite_blob(bench_doppler_effect(seed))
