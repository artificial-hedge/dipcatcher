"""Wave-616 tensor-category-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.braided_functor import bench_braided_functor
from quant_fund.models.center_cat import bench_center_cat
from quant_fund.models.ds_category import bench_ds_category
from quant_fund.models.fusion_ring import bench_fusion_ring
from quant_fund.models.multifusion import bench_multifusion
from quant_fund.models.premodular2 import bench_premodular2

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


def bench_multifusion_family(
    seed: int = _SEED + 3602,
) -> dict[str, float]:
    return _floats(_finite_blob("multifusion", bench_multifusion(seed)))


def bench_premodular2_family(
    seed: int = _SEED + 3603,
) -> dict[str, float]:
    return _floats(_finite_blob("premodular2", bench_premodular2(seed)))


def bench_braided_functor_family(
    seed: int = _SEED + 3604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "braided_functor",
            bench_braided_functor(seed),
        )
    )


def bench_center_cat_family(
    seed: int = _SEED + 3605,
) -> dict[str, float]:
    return _floats(_finite_blob("center_cat", bench_center_cat(seed)))


def bench_fusion_ring_family(
    seed: int = _SEED + 3606,
) -> dict[str, float]:
    return _floats(_finite_blob("fusion_ring", bench_fusion_ring(seed)))


def bench_ds_category_family(
    seed: int = _SEED + 3607,
) -> dict[str, float]:
    return _floats(_finite_blob("ds_category", bench_ds_category(seed)))
