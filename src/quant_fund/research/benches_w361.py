"""Wave-361 algebraic-topology-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chain_homotopy import bench_chain_homotopy
from quant_fund.models.covering_lift import bench_covering_lift
from quant_fund.models.degree_map import bench_degree_map
from quant_fund.models.euler_homology import bench_euler_homology
from quant_fund.models.homotopy_pi1 import bench_homotopy_pi1
from quant_fund.models.simplicial_homology import bench_simplicial_homology

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


def bench_homotopy_pi1_family(seed: int = _SEED + 2073) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_pi1", bench_homotopy_pi1(seed)))


def bench_simplicial_homology_family(seed: int = _SEED + 2074) -> dict[str, float]:
    return _floats(_finite_blob("simplicial_homology", bench_simplicial_homology(seed)))


def bench_chain_homotopy_family(seed: int = _SEED + 2075) -> dict[str, float]:
    return _floats(_finite_blob("chain_homotopy", bench_chain_homotopy(seed)))


def bench_euler_homology_family(seed: int = _SEED + 2076) -> dict[str, float]:
    return _floats(_finite_blob("euler_homology", bench_euler_homology(seed)))


def bench_degree_map_family(seed: int = _SEED + 2077) -> dict[str, float]:
    return _floats(_finite_blob("degree_map", bench_degree_map(seed)))


def bench_covering_lift_family(seed: int = _SEED + 2078) -> dict[str, float]:
    return _floats(_finite_blob("covering_lift", bench_covering_lift(seed)))
