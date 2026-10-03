"""Wave-620 stacks-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.band_gerbe import bench_band_gerbe
from quant_fund.models.dm_stack2 import bench_dm_stack2
from quant_fund.models.gerbe2 import bench_gerbe2
from quant_fund.models.inertia_stack import bench_inertia_stack
from quant_fund.models.rigid_stack import bench_rigid_stack
from quant_fund.models.root_stack import bench_root_stack

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


def bench_gerbe2_family(seed: int = _SEED + 3626) -> dict[str, float]:
    return _floats(_finite_blob("gerbe2", bench_gerbe2(seed)))


def bench_band_gerbe_family(
    seed: int = _SEED + 3627,
) -> dict[str, float]:
    return _floats(_finite_blob("band_gerbe", bench_band_gerbe(seed)))


def bench_rigid_stack_family(
    seed: int = _SEED + 3628,
) -> dict[str, float]:
    return _floats(_finite_blob("rigid_stack", bench_rigid_stack(seed)))


def bench_dm_stack2_family(
    seed: int = _SEED + 3629,
) -> dict[str, float]:
    return _floats(_finite_blob("dm_stack2", bench_dm_stack2(seed)))


def bench_inertia_stack_family(
    seed: int = _SEED + 3630,
) -> dict[str, float]:
    return _floats(_finite_blob("inertia_stack", bench_inertia_stack(seed)))


def bench_root_stack_family(
    seed: int = _SEED + 3631,
) -> dict[str, float]:
    return _floats(_finite_blob("root_stack", bench_root_stack(seed)))
