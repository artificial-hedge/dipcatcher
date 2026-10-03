"""Wave-295 astronomy/orbital-mechanics canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gauss_iod import bench_gauss_iod
from quant_fund.models.kepler_solve import bench_kepler_solve
from quant_fund.models.lambert_problem import bench_lambert_problem
from quant_fund.models.orbit_maneuver import bench_orbit_maneuver
from quant_fund.models.orbital_elements import bench_orbital_elements
from quant_fund.models.tle_propagate import bench_tle_propagate

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_orbital_elements_family(seed: int = _SEED + 1676) -> dict[str, float]:
    return _floats(_finite_blob("orbital_elements", bench_orbital_elements(seed)))


def bench_kepler_solve_family(seed: int = _SEED + 1677) -> dict[str, float]:
    return _floats(_finite_blob("kepler_solve", bench_kepler_solve(seed)))


def bench_lambert_problem_family(seed: int = _SEED + 1678) -> dict[str, float]:
    return _floats(_finite_blob("lambert_problem", bench_lambert_problem(seed)))


def bench_tle_propagate_family(seed: int = _SEED + 1679) -> dict[str, float]:
    return _floats(_finite_blob("tle_propagate", bench_tle_propagate(seed)))


def bench_orbit_maneuver_family(seed: int = _SEED + 1680) -> dict[str, float]:
    return _floats(_finite_blob("orbit_maneuver", bench_orbit_maneuver(seed)))


def bench_gauss_iod_family(seed: int = _SEED + 1681) -> dict[str, float]:
    return _floats(_finite_blob("gauss_iod", bench_gauss_iod(seed)))
