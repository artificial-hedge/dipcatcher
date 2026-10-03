"""Wave-427 number-theory-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bsd_toy import bench_bsd_toy
from quant_fund.models.elliptic_height import bench_elliptic_height
from quant_fund.models.lseries_toy import bench_lseries_toy
from quant_fund.models.modularity_toy import bench_modularity_toy
from quant_fund.models.mordell_weil import bench_mordell_weil
from quant_fund.models.padic_integral import bench_padic_integral

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


def bench_elliptic_height_family(
    seed: int = _SEED + 2468,
) -> dict[str, float]:
    return _floats(_finite_blob("elliptic_height", bench_elliptic_height(seed)))


def bench_mordell_weil_family(
    seed: int = _SEED + 2469,
) -> dict[str, float]:
    return _floats(_finite_blob("mordell_weil", bench_mordell_weil(seed)))


def bench_lseries_toy_family(
    seed: int = _SEED + 2470,
) -> dict[str, float]:
    return _floats(_finite_blob("lseries_toy", bench_lseries_toy(seed)))


def bench_bsd_toy_family(
    seed: int = _SEED + 2471,
) -> dict[str, float]:
    return _floats(_finite_blob("bsd_toy", bench_bsd_toy(seed)))


def bench_modularity_toy_family(
    seed: int = _SEED + 2472,
) -> dict[str, float]:
    return _floats(_finite_blob("modularity_toy", bench_modularity_toy(seed)))


def bench_padic_integral_family(
    seed: int = _SEED + 2473,
) -> dict[str, float]:
    return _floats(_finite_blob("padic_integral", bench_padic_integral(seed)))
