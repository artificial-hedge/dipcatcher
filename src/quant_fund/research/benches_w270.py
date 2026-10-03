"""Wave-270 computational-physics-2 benches: MD, FDTD, LBM, MCMC, PIC, DMC."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dmc_solver import bench_dmc_solver
from quant_fund.models.fdtd_wave import bench_fdtd_wave
from quant_fund.models.ising_metro import bench_ising_metro
from quant_fund.models.lattice_boltzmann import bench_lattice_boltzmann
from quant_fund.models.lj_md import bench_lj_md
from quant_fund.models.pic_plasma import bench_pic_plasma

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


def bench_lj_md_family(seed: int = _SEED + 1470) -> dict[str, float]:
    return _floats(_finite_blob("lj_md", bench_lj_md(seed)))


def bench_fdtd_wave_family(seed: int = _SEED + 1471) -> dict[str, float]:
    return _floats(_finite_blob("fdtd_wave", bench_fdtd_wave(seed)))


def bench_lattice_boltzmann_family(seed: int = _SEED + 1472) -> dict[str, float]:
    return _floats(_finite_blob("lattice_boltzmann", bench_lattice_boltzmann(seed)))


def bench_ising_metro_family(seed: int = _SEED + 1473) -> dict[str, float]:
    return _floats(_finite_blob("ising_metro", bench_ising_metro(seed)))


def bench_pic_plasma_family(seed: int = _SEED + 1474) -> dict[str, float]:
    return _floats(_finite_blob("pic_plasma", bench_pic_plasma(seed)))


def bench_dmc_solver_family(seed: int = _SEED + 1475) -> dict[str, float]:
    return _floats(_finite_blob("dmc_solver", bench_dmc_solver(seed)))
