"""Wave-433 p-adic-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adic_space import bench_adic_space
from quant_fund.models.berkovich_space import (
    bench_berkovich_space,
)
from quant_fund.models.diamond_toy import bench_diamond_toy
from quant_fund.models.etale_ph2 import bench_etale_ph2
from quant_fund.models.perfectoid_space import (
    bench_perfectoid_space,
)
from quant_fund.models.rigid_analytic import bench_rigid_analytic

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


def bench_rigid_analytic_family(
    seed: int = _SEED + 2504,
) -> dict[str, float]:
    return _floats(_finite_blob("rigid_analytic", bench_rigid_analytic(seed)))


def bench_berkovich_space_family(
    seed: int = _SEED + 2505,
) -> dict[str, float]:
    return _floats(_finite_blob("berkovich_space", bench_berkovich_space(seed)))


def bench_perfectoid_space_family(
    seed: int = _SEED + 2506,
) -> dict[str, float]:
    return _floats(_finite_blob("perfectoid_space", bench_perfectoid_space(seed)))


def bench_adic_space_family(
    seed: int = _SEED + 2507,
) -> dict[str, float]:
    return _floats(_finite_blob("adic_space", bench_adic_space(seed)))


def bench_etale_ph2_family(
    seed: int = _SEED + 2508,
) -> dict[str, float]:
    return _floats(_finite_blob("etale_ph2", bench_etale_ph2(seed)))


def bench_diamond_toy_family(
    seed: int = _SEED + 2509,
) -> dict[str, float]:
    return _floats(_finite_blob("diamond_toy", bench_diamond_toy(seed)))
