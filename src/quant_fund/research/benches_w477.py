"""Wave-477 arithmetic-D-modules-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arithmetic_ht import bench_arithmetic_ht
from quant_fund.models.berthelo_crys import bench_berthelo_crys
from quant_fund.models.berthelot_rigid import bench_berthelot_rigid
from quant_fund.models.caro_dm import bench_caro_dm
from quant_fund.models.dagger_dm import bench_dagger_dm
from quant_fund.models.spencer_dm import bench_spencer_dm

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


def bench_dagger_dm_family(seed: int = _SEED + 2768) -> dict[str, float]:
    return _floats(_finite_blob("dagger_dm", bench_dagger_dm(seed)))


def bench_spencer_dm_family(seed: int = _SEED + 2769) -> dict[str, float]:
    return _floats(_finite_blob("spencer_dm", bench_spencer_dm(seed)))


def bench_caro_dm_family(seed: int = _SEED + 2770) -> dict[str, float]:
    return _floats(_finite_blob("caro_dm", bench_caro_dm(seed)))


def bench_berthelot_rigid_family(seed: int = _SEED + 2771) -> dict[str, float]:
    return _floats(_finite_blob("berthelot_rigid", bench_berthelot_rigid(seed)))


def bench_berthelo_crys_family(seed: int = _SEED + 2772) -> dict[str, float]:
    return _floats(_finite_blob("berthelo_crys", bench_berthelo_crys(seed)))


def bench_arithmetic_ht_family(seed: int = _SEED + 2773) -> dict[str, float]:
    return _floats(_finite_blob("arithmetic_ht", bench_arithmetic_ht(seed)))
