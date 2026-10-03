"""Wave-760 Brownian-motion bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cameron_martin import bench_cameron_martin
from quant_fund.models.doob_bm import bench_doob_bm
from quant_fund.models.gikhman_skorokhod import (
    bench_gikhman_skorokhod,
)
from quant_fund.models.ito_bm import bench_ito_bm
from quant_fund.models.levy_bm import bench_levy_bm
from quant_fund.models.wiener_bm import bench_wiener_bm

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


def bench_levy_bm_family(
    seed: int = _SEED + 14900,
) -> dict[str, float]:
    return _floats(_finite_blob("levy_bm", bench_levy_bm(seed)))


def bench_wiener_bm_family(
    seed: int = _SEED + 14901,
) -> dict[str, float]:
    return _floats(_finite_blob("wiener_bm", bench_wiener_bm(seed)))


def bench_doob_bm_family(
    seed: int = _SEED + 14902,
) -> dict[str, float]:
    return _floats(_finite_blob("doob_bm", bench_doob_bm(seed)))


def bench_ito_bm_family(
    seed: int = _SEED + 14903,
) -> dict[str, float]:
    return _floats(_finite_blob("ito_bm", bench_ito_bm(seed)))


def bench_cameron_martin_family(
    seed: int = _SEED + 14904,
) -> dict[str, float]:
    return _floats(_finite_blob("cameron_martin", bench_cameron_martin(seed)))


def bench_gikhman_skorokhod_family(
    seed: int = _SEED + 14905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gikhman_skorokhod",
            bench_gikhman_skorokhod(seed),
        )
    )
