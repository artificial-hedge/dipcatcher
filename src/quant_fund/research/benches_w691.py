"""Wave-691 category-18 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cat_fusion import bench_cat_fusion
from quant_fund.models.cat_pretopos import bench_cat_pretopos
from quant_fund.models.cat_ribbon import bench_cat_ribbon
from quant_fund.models.cat_semiadd import bench_cat_semiadd
from quant_fund.models.cat_semisimple import (
    bench_cat_semisimple,
)
from quant_fund.models.cat_tannakian2 import (
    bench_cat_tannakian2,
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


def bench_cat_pretopos_family(
    seed: int = _SEED + 8000,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_pretopos", bench_cat_pretopos(seed)))


def bench_cat_semisimple_family(
    seed: int = _SEED + 8001,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_semisimple", bench_cat_semisimple(seed)))


def bench_cat_fusion_family(
    seed: int = _SEED + 8002,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_fusion", bench_cat_fusion(seed)))


def bench_cat_tannakian2_family(
    seed: int = _SEED + 8003,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_tannakian2", bench_cat_tannakian2(seed)))


def bench_cat_ribbon_family(
    seed: int = _SEED + 8004,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_ribbon", bench_cat_ribbon(seed)))


def bench_cat_semiadd_family(
    seed: int = _SEED + 8005,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_semiadd", bench_cat_semiadd(seed)))
