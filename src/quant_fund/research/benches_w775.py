"""Wave-775 martingale-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.burkholder_davis import (
    bench_burkholder_davis,
)
from quant_fund.models.doleans_meas import bench_doleans_meas
from quant_fund.models.gundy_mart import bench_gundy_mart
from quant_fund.models.local_mart import bench_local_mart
from quant_fund.models.predictable_proc import (
    bench_predictable_proc,
)
from quant_fund.models.square_bracket import (
    bench_square_bracket,
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


def bench_doleans_meas_family(
    seed: int = _SEED + 16400,
) -> dict[str, float]:
    return _floats(_finite_blob("doleans_meas", bench_doleans_meas(seed)))


def bench_predictable_proc_family(
    seed: int = _SEED + 16401,
) -> dict[str, float]:
    return _floats(_finite_blob("predictable_proc", bench_predictable_proc(seed)))


def bench_local_mart_family(
    seed: int = _SEED + 16402,
) -> dict[str, float]:
    return _floats(_finite_blob("local_mart", bench_local_mart(seed)))


def bench_square_bracket_family(
    seed: int = _SEED + 16403,
) -> dict[str, float]:
    return _floats(_finite_blob("square_bracket", bench_square_bracket(seed)))


def bench_burkholder_davis_family(
    seed: int = _SEED + 16404,
) -> dict[str, float]:
    return _floats(_finite_blob("burkholder_davis", bench_burkholder_davis(seed)))


def bench_gundy_mart_family(
    seed: int = _SEED + 16405,
) -> dict[str, float]:
    return _floats(_finite_blob("gundy_mart", bench_gundy_mart(seed)))
