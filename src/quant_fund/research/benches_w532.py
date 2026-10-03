"""Wave-532 fractal-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.box_counting import bench_box_counting
from quant_fund.models.frostman import bench_frostman
from quant_fund.models.hausdorff_dim import bench_hausdorff_dim
from quant_fund.models.iterated_function import bench_iterated_function
from quant_fund.models.multifractal_formal import (
    bench_multifractal_formal,
)
from quant_fund.models.self_similar import bench_self_similar

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


def bench_hausdorff_dim_family(seed: int = _SEED + 3098) -> dict[str, float]:
    return _floats(_finite_blob("hausdorff_dim", bench_hausdorff_dim(seed)))


def bench_box_counting_family(seed: int = _SEED + 3099) -> dict[str, float]:
    return _floats(_finite_blob("box_counting", bench_box_counting(seed)))


def bench_self_similar_family(seed: int = _SEED + 3100) -> dict[str, float]:
    return _floats(_finite_blob("self_similar", bench_self_similar(seed)))


def bench_iterated_function_family(
    seed: int = _SEED + 3101,
) -> dict[str, float]:
    return _floats(_finite_blob("iterated_function", bench_iterated_function(seed)))


def bench_frostman_family(seed: int = _SEED + 3102) -> dict[str, float]:
    return _floats(_finite_blob("frostman", bench_frostman(seed)))


def bench_multifractal_formal_family(
    seed: int = _SEED + 3103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "multifractal_formal",
            bench_multifractal_formal(seed),
        )
    )
