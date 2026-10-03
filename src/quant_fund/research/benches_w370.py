"""Wave-370 graph-theory-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dirac_ore import bench_dirac_ore
from quant_fund.models.graph_minor import bench_graph_minor
from quant_fund.models.planar_five import bench_planar_five
from quant_fund.models.ramsey_num import bench_ramsey_num
from quant_fund.models.turan_theorem import bench_turan_theorem
from quant_fund.models.tutte_berge import bench_tutte_berge

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


def bench_tutte_berge_family(seed: int = _SEED + 2126) -> dict[str, float]:
    return _floats(_finite_blob("tutte_berge", bench_tutte_berge(seed)))


def bench_dirac_ore_family(seed: int = _SEED + 2127) -> dict[str, float]:
    return _floats(_finite_blob("dirac_ore", bench_dirac_ore(seed)))


def bench_turan_theorem_family(seed: int = _SEED + 2128) -> dict[str, float]:
    return _floats(_finite_blob("turan_theorem", bench_turan_theorem(seed)))


def bench_planar_five_family(seed: int = _SEED + 2129) -> dict[str, float]:
    return _floats(_finite_blob("planar_five", bench_planar_five(seed)))


def bench_graph_minor_family(seed: int = _SEED + 2130) -> dict[str, float]:
    return _floats(_finite_blob("graph_minor", bench_graph_minor(seed)))


def bench_ramsey_num_family(seed: int = _SEED + 2131) -> dict[str, float]:
    return _floats(_finite_blob("ramsey_num", bench_ramsey_num(seed)))
