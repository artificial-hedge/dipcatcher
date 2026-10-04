"""Wave-592 tensor-category bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.braided_cat import bench_braided_cat
from quant_fund.models.fusion_cat import bench_fusion_cat
from quant_fund.models.premodular import bench_premodular
from quant_fund.models.rigid_cat import bench_rigid_cat
from quant_fund.models.spherical_cat import (
    bench_spherical_cat,
)
from quant_fund.models.tensor_cat import bench_tensor_cat

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


def bench_tensor_cat_family(
    seed: int = _SEED + 3458,
) -> dict[str, float]:
    return _floats(_finite_blob("tensor_cat", bench_tensor_cat(seed)))


def bench_braided_cat_family(
    seed: int = _SEED + 3459,
) -> dict[str, float]:
    return _floats(_finite_blob("braided_cat", bench_braided_cat(seed)))


def bench_rigid_cat_family(seed: int = _SEED + 3460) -> dict[str, float]:
    return _floats(_finite_blob("rigid_cat", bench_rigid_cat(seed)))


def bench_fusion_cat_family(
    seed: int = _SEED + 3461,
) -> dict[str, float]:
    return _floats(_finite_blob("fusion_cat", bench_fusion_cat(seed)))


def bench_spherical_cat_family(
    seed: int = _SEED + 3462,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spherical_cat",
            bench_spherical_cat(seed),
        )
    )


def bench_premodular_family(
    seed: int = _SEED + 3463,
) -> dict[str, float]:
    return _floats(_finite_blob("premodular", bench_premodular(seed)))
