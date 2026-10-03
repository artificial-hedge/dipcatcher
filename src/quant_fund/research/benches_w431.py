"""Wave-431 motivic-homotopy bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.a1_homotopy import bench_a1_homotopy
from quant_fund.models.milnor_operations import (
    bench_milnor_operations,
)
from quant_fund.models.morel_degree import bench_morel_degree
from quant_fund.models.motivic_sphere import bench_motivic_sphere
from quant_fund.models.slice_filtration import (
    bench_slice_filtration,
)
from quant_fund.models.voevodsky_motive import (
    bench_voevodsky_motive,
)

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


def bench_a1_homotopy_family(
    seed: int = _SEED + 2492,
) -> dict[str, float]:
    return _floats(_finite_blob("a1_homotopy", bench_a1_homotopy(seed)))


def bench_motivic_sphere_family(
    seed: int = _SEED + 2493,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_sphere", bench_motivic_sphere(seed)))


def bench_morel_degree_family(
    seed: int = _SEED + 2494,
) -> dict[str, float]:
    return _floats(_finite_blob("morel_degree", bench_morel_degree(seed)))


def bench_voevodsky_motive_family(
    seed: int = _SEED + 2495,
) -> dict[str, float]:
    return _floats(_finite_blob("voevodsky_motive", bench_voevodsky_motive(seed)))


def bench_slice_filtration_family(
    seed: int = _SEED + 2496,
) -> dict[str, float]:
    return _floats(_finite_blob("slice_filtration", bench_slice_filtration(seed)))


def bench_milnor_operations_family(
    seed: int = _SEED + 2497,
) -> dict[str, float]:
    return _floats(_finite_blob("milnor_operations", bench_milnor_operations(seed)))
