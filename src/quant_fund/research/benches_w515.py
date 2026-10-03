"""Wave-515 syzygy-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.auslander_buchs import bench_auslander_buchs
from quant_fund.models.betti_series import bench_betti_series
from quant_fund.models.green_koszul import bench_green_koszul
from quant_fund.models.minimal_free import bench_minimal_free
from quant_fund.models.quillen_suslin import bench_quillen_suslin
from quant_fund.models.serre_conj import bench_serre_conj

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


def bench_betti_series_family(seed: int = _SEED + 2996) -> dict[str, float]:
    return _floats(_finite_blob("betti_series", bench_betti_series(seed)))


def bench_minimal_free_family(seed: int = _SEED + 2997) -> dict[str, float]:
    return _floats(_finite_blob("minimal_free", bench_minimal_free(seed)))


def bench_auslander_buchs_family(seed: int = _SEED + 2998) -> dict[str, float]:
    return _floats(_finite_blob("auslander_buchs", bench_auslander_buchs(seed)))


def bench_serre_conj_family(seed: int = _SEED + 2999) -> dict[str, float]:
    return _floats(_finite_blob("serre_conj", bench_serre_conj(seed)))


def bench_quillen_suslin_family(seed: int = _SEED + 3000) -> dict[str, float]:
    return _floats(_finite_blob("quillen_suslin", bench_quillen_suslin(seed)))


def bench_green_koszul_family(seed: int = _SEED + 3001) -> dict[str, float]:
    return _floats(_finite_blob("green_koszul", bench_green_koszul(seed)))
