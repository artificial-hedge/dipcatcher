"""Wave-744 KPZ bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.amir_corwin import bench_amir_corwin
from quant_fund.models.borodin_corwin import bench_borodin_corwin
from quant_fund.models.calabrese_kpz import bench_calabrese_kpz
from quant_fund.models.corwin_kpz import bench_corwin_kpz
from quant_fund.models.kardar_parisi import bench_kardar_parisi
from quant_fund.models.quastel_spohn import bench_quastel_spohn

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


def bench_kardar_parisi_family(
    seed: int = _SEED + 13300,
) -> dict[str, float]:
    return _floats(_finite_blob("kardar_parisi", bench_kardar_parisi(seed)))


def bench_corwin_kpz_family(
    seed: int = _SEED + 13301,
) -> dict[str, float]:
    return _floats(_finite_blob("corwin_kpz", bench_corwin_kpz(seed)))


def bench_quastel_spohn_family(
    seed: int = _SEED + 13302,
) -> dict[str, float]:
    return _floats(_finite_blob("quastel_spohn", bench_quastel_spohn(seed)))


def bench_borodin_corwin_family(
    seed: int = _SEED + 13303,
) -> dict[str, float]:
    return _floats(_finite_blob("borodin_corwin", bench_borodin_corwin(seed)))


def bench_amir_corwin_family(
    seed: int = _SEED + 13304,
) -> dict[str, float]:
    return _floats(_finite_blob("amir_corwin", bench_amir_corwin(seed)))


def bench_calabrese_kpz_family(
    seed: int = _SEED + 13305,
) -> dict[str, float]:
    return _floats(_finite_blob("calabrese_kpz", bench_calabrese_kpz(seed)))
