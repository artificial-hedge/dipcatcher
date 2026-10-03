"""Wave-378 set-theory-2/forcing canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cohen_adds import bench_cohen_adds
from quant_fund.models.dense_filter import bench_dense_filter
from quant_fund.models.forcing_poset import bench_forcing_poset
from quant_fund.models.large_cardinal import bench_large_cardinal
from quant_fund.models.ma_toy import bench_ma_toy
from quant_fund.models.names_eval import bench_names_eval

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


def bench_forcing_poset_family(seed: int = _SEED + 2174) -> dict[str, float]:
    return _floats(_finite_blob("forcing_poset", bench_forcing_poset(seed)))


def bench_dense_filter_family(seed: int = _SEED + 2175) -> dict[str, float]:
    return _floats(_finite_blob("dense_filter", bench_dense_filter(seed)))


def bench_names_eval_family(seed: int = _SEED + 2176) -> dict[str, float]:
    return _floats(_finite_blob("names_eval", bench_names_eval(seed)))


def bench_cohen_adds_family(seed: int = _SEED + 2177) -> dict[str, float]:
    return _floats(_finite_blob("cohen_adds", bench_cohen_adds(seed)))


def bench_ma_toy_family(seed: int = _SEED + 2178) -> dict[str, float]:
    return _floats(_finite_blob("ma_toy", bench_ma_toy(seed)))


def bench_large_cardinal_family(seed: int = _SEED + 2179) -> dict[str, float]:
    return _floats(_finite_blob("large_cardinal", bench_large_cardinal(seed)))
