"""Wave-755 loop-soup bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.barlow_ust import bench_barlow_ust
from quant_fund.models.kassel_wu import bench_kassel_wu
from quant_fund.models.kenyon_wilson import bench_kenyon_wilson
from quant_fund.models.lejan_loop import bench_lejan_loop
from quant_fund.models.lupu_loop import bench_lupu_loop
from quant_fund.models.lyons_peres import bench_lyons_peres

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


def bench_lupu_loop_family(
    seed: int = _SEED + 14400,
) -> dict[str, float]:
    return _floats(_finite_blob("lupu_loop", bench_lupu_loop(seed)))


def bench_lejan_loop_family(
    seed: int = _SEED + 14401,
) -> dict[str, float]:
    return _floats(_finite_blob("lejan_loop", bench_lejan_loop(seed)))


def bench_kassel_wu_family(
    seed: int = _SEED + 14402,
) -> dict[str, float]:
    return _floats(_finite_blob("kassel_wu", bench_kassel_wu(seed)))


def bench_kenyon_wilson_family(
    seed: int = _SEED + 14403,
) -> dict[str, float]:
    return _floats(_finite_blob("kenyon_wilson", bench_kenyon_wilson(seed)))


def bench_barlow_ust_family(
    seed: int = _SEED + 14404,
) -> dict[str, float]:
    return _floats(_finite_blob("barlow_ust", bench_barlow_ust(seed)))


def bench_lyons_peres_family(
    seed: int = _SEED + 14405,
) -> dict[str, float]:
    return _floats(_finite_blob("lyons_peres", bench_lyons_peres(seed)))
