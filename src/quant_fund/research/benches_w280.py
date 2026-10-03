"""Wave-280 algebraic-topology benches: homology, Euler, Rips, degree."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.boundary_sq import bench_boundary_sq
from quant_fund.models.euler_char import bench_euler_char
from quant_fund.models.graph_h1 import bench_graph_h1
from quant_fund.models.rips_h1 import bench_rips_h1
from quant_fund.models.simp_betti import bench_simp_betti
from quant_fund.models.winding_deg import bench_winding_deg

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_simp_betti_family(seed: int = _SEED + 1570) -> dict[str, float]:
    return _floats(_finite_blob("simp_betti", bench_simp_betti(seed)))


def bench_boundary_sq_family(seed: int = _SEED + 1571) -> dict[str, float]:
    return _floats(_finite_blob("boundary_sq", bench_boundary_sq(seed)))


def bench_euler_char_family(seed: int = _SEED + 1572) -> dict[str, float]:
    return _floats(_finite_blob("euler_char", bench_euler_char(seed)))


def bench_rips_h1_family(seed: int = _SEED + 1573) -> dict[str, float]:
    return _floats(_finite_blob("rips_h1", bench_rips_h1(seed)))


def bench_graph_h1_family(seed: int = _SEED + 1574) -> dict[str, float]:
    return _floats(_finite_blob("graph_h1", bench_graph_h1(seed)))


def bench_winding_deg_family(seed: int = _SEED + 1575) -> dict[str, float]:
    return _floats(_finite_blob("winding_deg", bench_winding_deg(seed)))
