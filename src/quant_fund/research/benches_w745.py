"""Wave-745 KPZ-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bernard_nicola import bench_bernard_nicola
from quant_fund.models.dotsenko_kpz import bench_dotsenko_kpz
from quant_fund.models.hairer_kpz import bench_hairer_kpz
from quant_fund.models.imamura_sasamoto import (
    bench_imamura_sasamoto,
)
from quant_fund.models.spohn_kpz import bench_spohn_kpz
from quant_fund.models.tracy_widom_kpz import bench_tracy_widom_kpz

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


def bench_dotsenko_kpz_family(
    seed: int = _SEED + 13400,
) -> dict[str, float]:
    return _floats(_finite_blob("dotsenko_kpz", bench_dotsenko_kpz(seed)))


def bench_hairer_kpz_family(
    seed: int = _SEED + 13401,
) -> dict[str, float]:
    return _floats(_finite_blob("hairer_kpz", bench_hairer_kpz(seed)))


def bench_bernard_nicola_family(
    seed: int = _SEED + 13402,
) -> dict[str, float]:
    return _floats(_finite_blob("bernard_nicola", bench_bernard_nicola(seed)))


def bench_imamura_sasamoto_family(
    seed: int = _SEED + 13403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "imamura_sasamoto",
            bench_imamura_sasamoto(seed),
        )
    )


def bench_tracy_widom_kpz_family(
    seed: int = _SEED + 13404,
) -> dict[str, float]:
    return _floats(_finite_blob("tracy_widom_kpz", bench_tracy_widom_kpz(seed)))


def bench_spohn_kpz_family(
    seed: int = _SEED + 13405,
) -> dict[str, float]:
    return _floats(_finite_blob("spohn_kpz", bench_spohn_kpz(seed)))
