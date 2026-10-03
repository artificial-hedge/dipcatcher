"""Wave-347 topology-3/point-set canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.compact_space import bench_compact_space
from quant_fund.models.connected_space import bench_connected_space
from quant_fund.models.convergence_space import bench_convergence_space
from quant_fund.models.product_topology import bench_product_topology
from quant_fund.models.quotient_topology import bench_quotient_topology
from quant_fund.models.tietze_urysohn import bench_tietze_urysohn

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


def bench_compact_space_family(seed: int = _SEED + 1989) -> dict[str, float]:
    return _floats(_finite_blob("compact_space", bench_compact_space(seed)))


def bench_connected_space_family(seed: int = _SEED + 1990) -> dict[str, float]:
    return _floats(_finite_blob("connected_space", bench_connected_space(seed)))


def bench_quotient_topology_family(seed: int = _SEED + 1991) -> dict[str, float]:
    return _floats(_finite_blob("quotient_topology", bench_quotient_topology(seed)))


def bench_product_topology_family(seed: int = _SEED + 1992) -> dict[str, float]:
    return _floats(_finite_blob("product_topology", bench_product_topology(seed)))


def bench_convergence_space_family(seed: int = _SEED + 1993) -> dict[str, float]:
    return _floats(_finite_blob("convergence_space", bench_convergence_space(seed)))


def bench_tietze_urysohn_family(seed: int = _SEED + 1994) -> dict[str, float]:
    return _floats(_finite_blob("tietze_urysohn", bench_tietze_urysohn(seed)))
