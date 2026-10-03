"""Wave-380 descriptive-set-theory-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.baire_space import bench_baire_space
from quant_fund.models.borel_functions import bench_borel_functions
from quant_fund.models.determinacy_toy import bench_determinacy_toy
from quant_fund.models.perfect_set_prop import bench_perfect_set_prop
from quant_fund.models.polish_topology import bench_polish_topology
from quant_fund.models.souslin_op import bench_souslin_op

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


def bench_baire_space_family(seed: int = _SEED + 2186) -> dict[str, float]:
    return _floats(_finite_blob("baire_space", bench_baire_space(seed)))


def bench_polish_topology_family(seed: int = _SEED + 2187) -> dict[str, float]:
    return _floats(_finite_blob("polish_topology", bench_polish_topology(seed)))


def bench_borel_functions_family(seed: int = _SEED + 2188) -> dict[str, float]:
    return _floats(_finite_blob("borel_functions", bench_borel_functions(seed)))


def bench_souslin_op_family(seed: int = _SEED + 2189) -> dict[str, float]:
    return _floats(_finite_blob("souslin_op", bench_souslin_op(seed)))


def bench_determinacy_toy_family(seed: int = _SEED + 2190) -> dict[str, float]:
    return _floats(_finite_blob("determinacy_toy", bench_determinacy_toy(seed)))


def bench_perfect_set_prop_family(seed: int = _SEED + 2191) -> dict[str, float]:
    return _floats(_finite_blob("perfect_set_prop", bench_perfect_set_prop(seed)))
