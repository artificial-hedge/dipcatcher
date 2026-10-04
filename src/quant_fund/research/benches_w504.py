"""Wave-504 Steenrod/cohomology-operations bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adem_relations import bench_adem_relations
from quant_fund.models.bar_resolution import bench_bar_resolution
from quant_fund.models.lambda_algebra import bench_lambda_algebra
from quant_fund.models.serre_cartan import bench_serre_cartan
from quant_fund.models.steenrod_algebra import bench_steenrod_algebra
from quant_fund.models.unstable_modules import bench_unstable_modules

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


def bench_steenrod_algebra_family(seed: int = _SEED + 2930) -> dict[str, float]:
    return _floats(_finite_blob("steenrod_algebra", bench_steenrod_algebra(seed)))


def bench_adem_relations_family(seed: int = _SEED + 2931) -> dict[str, float]:
    return _floats(_finite_blob("adem_relations", bench_adem_relations(seed)))


def bench_serre_cartan_family(seed: int = _SEED + 2932) -> dict[str, float]:
    return _floats(_finite_blob("serre_cartan", bench_serre_cartan(seed)))


def bench_unstable_modules_family(seed: int = _SEED + 2933) -> dict[str, float]:
    return _floats(_finite_blob("unstable_modules", bench_unstable_modules(seed)))


def bench_lambda_algebra_family(seed: int = _SEED + 2934) -> dict[str, float]:
    return _floats(_finite_blob("lambda_algebra", bench_lambda_algebra(seed)))


def bench_bar_resolution_family(seed: int = _SEED + 2935) -> dict[str, float]:
    return _floats(_finite_blob("bar_resolution", bench_bar_resolution(seed)))
