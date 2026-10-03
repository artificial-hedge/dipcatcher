"""Wave-225 adapters: physics-simulation canon — nbody_leapfrog, barnes_hut,
sph_fluid, rigid_collision, verlet_cloth, fem_truss —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.barnes_hut import bench_barnes_hut
from quant_fund.models.fem_truss import bench_fem_truss
from quant_fund.models.nbody_leapfrog import bench_nbody_leapfrog
from quant_fund.models.rigid_collision import bench_rigid_collision
from quant_fund.models.sph_fluid import bench_sph_fluid
from quant_fund.models.verlet_cloth import bench_verlet_cloth

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


def bench_barnes_hut_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("barnes_hut", bench_barnes_hut(seed=_SEED + 1020)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"barnes_hut bench failed: {exc}") from exc


def bench_fem_truss_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fem_truss", bench_fem_truss(seed=_SEED + 1021)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fem_truss bench failed: {exc}") from exc


def bench_nbody_leapfrog_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nbody_leapfrog", bench_nbody_leapfrog(seed=_SEED + 1022)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nbody_leapfrog bench failed: {exc}") from exc


def bench_rigid_collision_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rigid_collision", bench_rigid_collision(seed=_SEED + 1023)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rigid_collision bench failed: {exc}") from exc


def bench_sph_fluid_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sph_fluid", bench_sph_fluid(seed=_SEED + 1024)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sph_fluid bench failed: {exc}") from exc


def bench_verlet_cloth_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("verlet_cloth", bench_verlet_cloth(seed=_SEED + 1025)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"verlet_cloth bench failed: {exc}") from exc
