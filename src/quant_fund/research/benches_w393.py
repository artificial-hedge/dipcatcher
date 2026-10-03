"""Wave-393 algebraic-topology-4 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cap_product import bench_cap_product
from quant_fund.models.eilenberg_steenrod import bench_eilenberg_steenrod
from quant_fund.models.k_theory import bench_k_theory
from quant_fund.models.obstruction_toy import bench_obstruction_toy
from quant_fund.models.serre_class import bench_serre_class
from quant_fund.models.thom_isom import bench_thom_isom

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


def bench_eilenberg_steenrod_family(
    seed: int = _SEED + 2264,
) -> dict[str, float]:
    return _floats(_finite_blob("eilenberg_steenrod", bench_eilenberg_steenrod(seed)))


def bench_cap_product_family(seed: int = _SEED + 2265) -> dict[str, float]:
    return _floats(_finite_blob("cap_product", bench_cap_product(seed)))


def bench_thom_isom_family(seed: int = _SEED + 2266) -> dict[str, float]:
    return _floats(_finite_blob("thom_isom", bench_thom_isom(seed)))


def bench_serre_class_family(seed: int = _SEED + 2267) -> dict[str, float]:
    return _floats(_finite_blob("serre_class", bench_serre_class(seed)))


def bench_obstruction_toy_family(
    seed: int = _SEED + 2268,
) -> dict[str, float]:
    return _floats(_finite_blob("obstruction_toy", bench_obstruction_toy(seed)))


def bench_k_theory_family(seed: int = _SEED + 2269) -> dict[str, float]:
    return _floats(_finite_blob("k_theory", bench_k_theory(seed)))
