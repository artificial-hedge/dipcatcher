"""Wave-523 incidence-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.erdos_distinct import bench_erdos_distinct
from quant_fund.models.ff_kakeya import bench_ff_kakeya
from quant_fund.models.guth_katz import bench_guth_katz
from quant_fund.models.joints_thm import bench_joints_thm
from quant_fund.models.kakeya import bench_kakeya
from quant_fund.models.sz_trotter import bench_sz_trotter

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


def bench_erdos_distinct_family(seed: int = _SEED + 3044) -> dict[str, float]:
    return _floats(_finite_blob("erdos_distinct", bench_erdos_distinct(seed)))


def bench_sz_trotter_family(seed: int = _SEED + 3045) -> dict[str, float]:
    return _floats(_finite_blob("sz_trotter", bench_sz_trotter(seed)))


def bench_kakeya_family(seed: int = _SEED + 3046) -> dict[str, float]:
    return _floats(_finite_blob("kakeya", bench_kakeya(seed)))


def bench_ff_kakeya_family(seed: int = _SEED + 3047) -> dict[str, float]:
    return _floats(_finite_blob("ff_kakeya", bench_ff_kakeya(seed)))


def bench_joints_thm_family(seed: int = _SEED + 3048) -> dict[str, float]:
    return _floats(_finite_blob("joints_thm", bench_joints_thm(seed)))


def bench_guth_katz_family(seed: int = _SEED + 3049) -> dict[str, float]:
    return _floats(_finite_blob("guth_katz", bench_guth_katz(seed)))
