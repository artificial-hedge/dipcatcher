"""Wave-1141 astronomy-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.astrobiology import bench_astrobiology
from quant_fund.models.astrochemistry import bench_astrochemistry
from quant_fund.models.cosmology_2 import bench_cosmology_2
from quant_fund.models.exoplanet_science import bench_exoplanet_science
from quant_fund.models.galactic_dynamics import bench_galactic_dynamics
from quant_fund.models.helio_seismology import bench_helio_seismology

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


def bench_cosmology_2_family(seed: int = _SEED + 52900) -> dict[str, float]:
    return _finite_blob(bench_cosmology_2(seed))


def bench_astrobiology_family(seed: int = _SEED + 52901) -> dict[str, float]:
    return _finite_blob(bench_astrobiology(seed))


def bench_astrochemistry_family(seed: int = _SEED + 52902) -> dict[str, float]:
    return _finite_blob(bench_astrochemistry(seed))


def bench_helio_seismology_family(seed: int = _SEED + 52903) -> dict[str, float]:
    return _finite_blob(bench_helio_seismology(seed))


def bench_exoplanet_science_family(seed: int = _SEED + 52904) -> dict[str, float]:
    return _finite_blob(bench_exoplanet_science(seed))


def bench_galactic_dynamics_family(seed: int = _SEED + 52905) -> dict[str, float]:
    return _finite_blob(bench_galactic_dynamics(seed))
