"""Wave-344 group-theory-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.burnside_lemma import bench_burnside_lemma
from quant_fund.models.cayley_graph import bench_cayley_graph
from quant_fund.models.conjugacy_classes import bench_conjugacy_classes
from quant_fund.models.free_group import bench_free_group
from quant_fund.models.group_presentation import bench_group_presentation
from quant_fund.models.sylow_theorems import bench_sylow_theorems

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


def bench_sylow_theorems_family(seed: int = _SEED + 1971) -> dict[str, float]:
    return _floats(_finite_blob("sylow_theorems", bench_sylow_theorems(seed)))


def bench_group_presentation_family(seed: int = _SEED + 1972) -> dict[str, float]:
    return _floats(_finite_blob("group_presentation", bench_group_presentation(seed)))


def bench_burnside_lemma_family(seed: int = _SEED + 1973) -> dict[str, float]:
    return _floats(_finite_blob("burnside_lemma", bench_burnside_lemma(seed)))


def bench_free_group_family(seed: int = _SEED + 1974) -> dict[str, float]:
    return _floats(_finite_blob("free_group", bench_free_group(seed)))


def bench_conjugacy_classes_family(seed: int = _SEED + 1975) -> dict[str, float]:
    return _floats(_finite_blob("conjugacy_classes", bench_conjugacy_classes(seed)))


def bench_cayley_graph_family(seed: int = _SEED + 1976) -> dict[str, float]:
    return _floats(_finite_blob("cayley_graph", bench_cayley_graph(seed)))
