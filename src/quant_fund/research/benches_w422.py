"""Wave-422 Lie-theory-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.borel_subalgebra import bench_borel_subalgebra
from quant_fund.models.levi_factor import bench_levi_factor
from quant_fund.models.nilpotent_orbit import bench_nilpotent_orbit
from quant_fund.models.root_height import bench_root_height
from quant_fund.models.verma_module import bench_verma_module
from quant_fund.models.weyl_chamber import bench_weyl_chamber

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


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


def bench_weyl_chamber_family(
    seed: int = _SEED + 2438,
) -> dict[str, float]:
    return _floats(_finite_blob("weyl_chamber", bench_weyl_chamber(seed)))


def bench_root_height_family(
    seed: int = _SEED + 2439,
) -> dict[str, float]:
    return _floats(_finite_blob("root_height", bench_root_height(seed)))


def bench_borel_subalgebra_family(
    seed: int = _SEED + 2440,
) -> dict[str, float]:
    return _floats(_finite_blob("borel_subalgebra", bench_borel_subalgebra(seed)))


def bench_levi_factor_family(
    seed: int = _SEED + 2441,
) -> dict[str, float]:
    return _floats(_finite_blob("levi_factor", bench_levi_factor(seed)))


def bench_nilpotent_orbit_family(
    seed: int = _SEED + 2442,
) -> dict[str, float]:
    return _floats(_finite_blob("nilpotent_orbit", bench_nilpotent_orbit(seed)))


def bench_verma_module_family(
    seed: int = _SEED + 2443,
) -> dict[str, float]:
    return _floats(_finite_blob("verma_module", bench_verma_module(seed)))
