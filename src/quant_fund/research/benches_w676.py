"""Wave-676 higher-algebra-7 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.center_hochschild import (
    bench_center_hochschild,
)
from quant_fund.models.en_algebra2 import bench_en_algebra2
from quant_fund.models.higher_brace2 import bench_higher_brace2
from quant_fund.models.koszul_operad2 import bench_koszul_operad2
from quant_fund.models.operad_lie import bench_operad_lie
from quant_fund.models.thom_transpose import bench_thom_transpose

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


def bench_en_algebra2_family(
    seed: int = _SEED + 6500,
) -> dict[str, float]:
    return _floats(_finite_blob("en_algebra2", bench_en_algebra2(seed)))


def bench_thom_transpose_family(
    seed: int = _SEED + 6501,
) -> dict[str, float]:
    return _floats(_finite_blob("thom_transpose", bench_thom_transpose(seed)))


def bench_higher_brace2_family(
    seed: int = _SEED + 6502,
) -> dict[str, float]:
    return _floats(_finite_blob("higher_brace2", bench_higher_brace2(seed)))


def bench_koszul_operad2_family(
    seed: int = _SEED + 6503,
) -> dict[str, float]:
    return _floats(_finite_blob("koszul_operad2", bench_koszul_operad2(seed)))


def bench_operad_lie_family(
    seed: int = _SEED + 6504,
) -> dict[str, float]:
    return _floats(_finite_blob("operad_lie", bench_operad_lie(seed)))


def bench_center_hochschild_family(
    seed: int = _SEED + 6505,
) -> dict[str, float]:
    return _floats(_finite_blob("center_hochschild", bench_center_hochschild(seed)))
