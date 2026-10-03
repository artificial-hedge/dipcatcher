"""Wave-709 category-22 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cat_ab2 import bench_cat_ab2
from quant_fund.models.cat_ab_loc import bench_cat_ab_loc
from quant_fund.models.cat_exact3 import bench_cat_exact3
from quant_fund.models.cat_freyd import bench_cat_freyd
from quant_fund.models.cat_pro_object2 import (
    bench_cat_pro_object2,
)
from quant_fund.models.cat_univariant2 import (
    bench_cat_univariant2,
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


def bench_cat_univariant2_family(
    seed: int = _SEED + 9800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cat_univariant2",
            bench_cat_univariant2(seed),
        )
    )


def bench_cat_ab2_family(
    seed: int = _SEED + 9801,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_ab2", bench_cat_ab2(seed)))


def bench_cat_exact3_family(
    seed: int = _SEED + 9802,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_exact3", bench_cat_exact3(seed)))


def bench_cat_freyd_family(
    seed: int = _SEED + 9803,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_freyd", bench_cat_freyd(seed)))


def bench_cat_ab_loc_family(
    seed: int = _SEED + 9804,
) -> dict[str, float]:
    return _floats(_finite_blob("cat_ab_loc", bench_cat_ab_loc(seed)))


def bench_cat_pro_object2_family(
    seed: int = _SEED + 9805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cat_pro_object2",
            bench_cat_pro_object2(seed),
        )
    )
