"""Wave-646 p-adic-7 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bdr_plus import bench_bdr_plus
from quant_fund.models.diamond_sheaf import (
    bench_diamond_sheaf,
)
from quant_fund.models.fargues_cat import bench_fargues_cat
from quant_fund.models.spatial_diamond import (
    bench_spatial_diamond,
)
from quant_fund.models.untilt2 import bench_untilt2
from quant_fund.models.v_stack import bench_v_stack

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


def bench_fargues_cat_family(
    seed: int = _SEED + 3782,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fargues_cat",
            bench_fargues_cat(seed),
        )
    )


def bench_v_stack_family(
    seed: int = _SEED + 3783,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "v_stack",
            bench_v_stack(seed),
        )
    )


def bench_untilt2_family(
    seed: int = _SEED + 3784,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "untilt2",
            bench_untilt2(seed),
        )
    )


def bench_spatial_diamond_family(
    seed: int = _SEED + 3785,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spatial_diamond",
            bench_spatial_diamond(seed),
        )
    )


def bench_diamond_sheaf_family(
    seed: int = _SEED + 3786,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "diamond_sheaf",
            bench_diamond_sheaf(seed),
        )
    )


def bench_bdr_plus_family(
    seed: int = _SEED + 3787,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bdr_plus",
            bench_bdr_plus(seed),
        )
    )
