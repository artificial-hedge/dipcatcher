"""Wave-634 category-9 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bicat2 import bench_bicat2
from quant_fund.models.cat_3cell import bench_cat_3cell
from quant_fund.models.double_lim import bench_double_lim
from quant_fund.models.icon_cat import bench_icon_cat
from quant_fund.models.two_transform import (
    bench_two_transform,
)
from quant_fund.models.vert_cat import bench_vert_cat

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


def bench_icon_cat_family(
    seed: int = _SEED + 3710,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "icon_cat",
            bench_icon_cat(seed),
        )
    )


def bench_bicat2_family(
    seed: int = _SEED + 3711,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bicat2",
            bench_bicat2(seed),
        )
    )


def bench_vert_cat_family(
    seed: int = _SEED + 3712,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "vert_cat",
            bench_vert_cat(seed),
        )
    )


def bench_double_lim_family(
    seed: int = _SEED + 3713,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "double_lim",
            bench_double_lim(seed),
        )
    )


def bench_two_transform_family(
    seed: int = _SEED + 3714,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "two_transform",
            bench_two_transform(seed),
        )
    )


def bench_cat_3cell_family(
    seed: int = _SEED + 3715,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cat_3cell",
            bench_cat_3cell(seed),
        )
    )
