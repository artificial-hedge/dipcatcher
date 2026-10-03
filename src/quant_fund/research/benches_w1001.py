"""Wave-1001 fluid-dynamics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.beale_kato_majda import bench_beale_kato_majda
from quant_fund.models.euler_equations import bench_euler_equations
from quant_fund.models.ladyzhenskaya_weak import bench_ladyzhenskaya_weak
from quant_fund.models.leray_theory import bench_leray_theory
from quant_fund.models.navier_stokes import bench_navier_stokes
from quant_fund.models.vorticity_form import bench_vorticity_form

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


def bench_euler_equations_family(seed: int = _SEED + 38900) -> dict[str, float]:
    return _finite_blob(bench_euler_equations(seed))


def bench_navier_stokes_family(seed: int = _SEED + 38901) -> dict[str, float]:
    return _finite_blob(bench_navier_stokes(seed))


def bench_vorticity_form_family(seed: int = _SEED + 38902) -> dict[str, float]:
    return _finite_blob(bench_vorticity_form(seed))


def bench_beale_kato_majda_family(seed: int = _SEED + 38903) -> dict[str, float]:
    return _finite_blob(bench_beale_kato_majda(seed))


def bench_ladyzhenskaya_weak_family(seed: int = _SEED + 38904) -> dict[str, float]:
    return _finite_blob(bench_ladyzhenskaya_weak(seed))


def bench_leray_theory_family(seed: int = _SEED + 38905) -> dict[str, float]:
    return _finite_blob(bench_leray_theory(seed))
