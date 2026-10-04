"""Wave-749 vertex-model-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.borodin_bufetov import (
    bench_borodin_bufetov,
)
from quant_fund.models.borodin_wheeler import (
    bench_borodin_wheeler,
)
from quant_fund.models.bufetov_sixv import bench_bufetov_sixv
from quant_fund.models.dimitrov_sixv import bench_dimitrov_sixv
from quant_fund.models.kuan_sixv import bench_kuan_sixv
from quant_fund.models.wheeler_zinn import bench_wheeler_zinn

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


def bench_bufetov_sixv_family(
    seed: int = _SEED + 13800,
) -> dict[str, float]:
    return _floats(_finite_blob("bufetov_sixv", bench_bufetov_sixv(seed)))


def bench_borodin_bufetov_family(
    seed: int = _SEED + 13801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "borodin_bufetov",
            bench_borodin_bufetov(seed),
        )
    )


def bench_kuan_sixv_family(
    seed: int = _SEED + 13802,
) -> dict[str, float]:
    return _floats(_finite_blob("kuan_sixv", bench_kuan_sixv(seed)))


def bench_dimitrov_sixv_family(
    seed: int = _SEED + 13803,
) -> dict[str, float]:
    return _floats(_finite_blob("dimitrov_sixv", bench_dimitrov_sixv(seed)))


def bench_borodin_wheeler_family(
    seed: int = _SEED + 13804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "borodin_wheeler",
            bench_borodin_wheeler(seed),
        )
    )


def bench_wheeler_zinn_family(
    seed: int = _SEED + 13805,
) -> dict[str, float]:
    return _floats(_finite_blob("wheeler_zinn", bench_wheeler_zinn(seed)))
