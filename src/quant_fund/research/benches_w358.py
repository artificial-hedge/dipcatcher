"""Wave-358 functional-analysis-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.banach_alaoglu import bench_banach_alaoglu
from quant_fund.models.closed_graph import bench_closed_graph
from quant_fund.models.open_mapping import bench_open_mapping
from quant_fund.models.reflexive_space import bench_reflexive_space
from quant_fund.models.uniform_bounded import bench_uniform_bounded
from quant_fund.models.weak_convergence import bench_weak_convergence

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


def bench_open_mapping_family(seed: int = _SEED + 2055) -> dict[str, float]:
    return _floats(_finite_blob("open_mapping", bench_open_mapping(seed)))


def bench_uniform_bounded_family(seed: int = _SEED + 2056) -> dict[str, float]:
    return _floats(_finite_blob("uniform_bounded", bench_uniform_bounded(seed)))


def bench_weak_convergence_family(seed: int = _SEED + 2057) -> dict[str, float]:
    return _floats(_finite_blob("weak_convergence", bench_weak_convergence(seed)))


def bench_banach_alaoglu_family(seed: int = _SEED + 2058) -> dict[str, float]:
    return _floats(_finite_blob("banach_alaoglu", bench_banach_alaoglu(seed)))


def bench_reflexive_space_family(seed: int = _SEED + 2059) -> dict[str, float]:
    return _floats(_finite_blob("reflexive_space", bench_reflexive_space(seed)))


def bench_closed_graph_family(seed: int = _SEED + 2060) -> dict[str, float]:
    return _floats(_finite_blob("closed_graph", bench_closed_graph(seed)))
