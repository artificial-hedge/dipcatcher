"""Wave-313 numerical-4/multigrid canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.amg_lite import bench_amg_lite
from quant_fund.models.bicgstab import bench_bicgstab
from quant_fund.models.chebyshev_iter import bench_chebyshev_iter
from quant_fund.models.ilu_precond import bench_ilu_precond
from quant_fund.models.minres import bench_minres
from quant_fund.models.v_cycle import bench_v_cycle

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


def bench_v_cycle_family(seed: int = _SEED + 1785) -> dict[str, float]:
    return _floats(_finite_blob("v_cycle", bench_v_cycle(seed)))


def bench_amg_lite_family(seed: int = _SEED + 1786) -> dict[str, float]:
    return _floats(_finite_blob("amg_lite", bench_amg_lite(seed)))


def bench_bicgstab_family(seed: int = _SEED + 1787) -> dict[str, float]:
    return _floats(_finite_blob("bicgstab", bench_bicgstab(seed)))


def bench_minres_family(seed: int = _SEED + 1788) -> dict[str, float]:
    return _floats(_finite_blob("minres", bench_minres(seed)))


def bench_chebyshev_iter_family(seed: int = _SEED + 1789) -> dict[str, float]:
    return _floats(_finite_blob("chebyshev_iter", bench_chebyshev_iter(seed)))


def bench_ilu_precond_family(seed: int = _SEED + 1790) -> dict[str, float]:
    return _floats(_finite_blob("ilu_precond", bench_ilu_precond(seed)))
