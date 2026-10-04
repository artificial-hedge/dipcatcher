"""Wave-321 shape-analysis canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.context_pta import bench_context_pta
from quant_fund.models.interproc_summary import bench_interproc_summary
from quant_fund.models.recency_abstraction import bench_recency_abstraction
from quant_fund.models.separation_logic import bench_separation_logic
from quant_fund.models.shape_graph import bench_shape_graph
from quant_fund.models.three_valued_logic import bench_three_valued_logic

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


def bench_three_valued_logic_family(seed: int = _SEED + 1833) -> dict[str, float]:
    return _floats(_finite_blob("three_valued_logic", bench_three_valued_logic(seed)))


def bench_shape_graph_family(seed: int = _SEED + 1834) -> dict[str, float]:
    return _floats(_finite_blob("shape_graph", bench_shape_graph(seed)))


def bench_separation_logic_family(seed: int = _SEED + 1835) -> dict[str, float]:
    return _floats(_finite_blob("separation_logic", bench_separation_logic(seed)))


def bench_context_pta_family(seed: int = _SEED + 1836) -> dict[str, float]:
    return _floats(_finite_blob("context_pta", bench_context_pta(seed)))


def bench_recency_abstraction_family(seed: int = _SEED + 1837) -> dict[str, float]:
    return _floats(_finite_blob("recency_abstraction", bench_recency_abstraction(seed)))


def bench_interproc_summary_family(seed: int = _SEED + 1838) -> dict[str, float]:
    return _floats(_finite_blob("interproc_summary", bench_interproc_summary(seed)))
