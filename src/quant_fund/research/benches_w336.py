"""Wave-336 computability/model-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.busy_beaver import bench_busy_beaver
from quant_fund.models.compactness_lite import bench_compactness_lite
from quant_fund.models.pr_functions import bench_pr_functions
from quant_fund.models.ramsey_theory import bench_ramsey_theory
from quant_fund.models.turing_degrees import bench_turing_degrees
from quant_fund.models.ultraproduct import bench_ultraproduct

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


def bench_pr_functions_family(seed: int = _SEED + 1923) -> dict[str, float]:
    return _floats(_finite_blob("pr_functions", bench_pr_functions(seed)))


def bench_turing_degrees_family(seed: int = _SEED + 1924) -> dict[str, float]:
    return _floats(_finite_blob("turing_degrees", bench_turing_degrees(seed)))


def bench_busy_beaver_family(seed: int = _SEED + 1925) -> dict[str, float]:
    return _floats(_finite_blob("busy_beaver", bench_busy_beaver(seed)))


def bench_ultraproduct_family(seed: int = _SEED + 1926) -> dict[str, float]:
    return _floats(_finite_blob("ultraproduct", bench_ultraproduct(seed)))


def bench_ramsey_theory_family(seed: int = _SEED + 1927) -> dict[str, float]:
    return _floats(_finite_blob("ramsey_theory", bench_ramsey_theory(seed)))


def bench_compactness_lite_family(seed: int = _SEED + 1928) -> dict[str, float]:
    return _floats(_finite_blob("compactness_lite", bench_compactness_lite(seed)))
