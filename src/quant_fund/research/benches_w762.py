"""Wave-762 weak-convergence bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cadlag_space import bench_cadlag_space
from quant_fund.models.doob_meyer import bench_doob_meyer
from quant_fund.models.martin_boundary import bench_martin_boundary
from quant_fund.models.prohorov_thm2 import bench_prohorov_thm2
from quant_fund.models.skohorod_metric import (
    bench_skohorod_metric,
)
from quant_fund.models.tightness_check import bench_tightness_check

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


def bench_martin_boundary_family(
    seed: int = _SEED + 15100,
) -> dict[str, float]:
    return _floats(_finite_blob("martin_boundary", bench_martin_boundary(seed)))


def bench_doob_meyer_family(
    seed: int = _SEED + 15101,
) -> dict[str, float]:
    return _floats(_finite_blob("doob_meyer", bench_doob_meyer(seed)))


def bench_cadlag_space_family(
    seed: int = _SEED + 15102,
) -> dict[str, float]:
    return _floats(_finite_blob("cadlag_space", bench_cadlag_space(seed)))


def bench_skohorod_metric_family(
    seed: int = _SEED + 15103,
) -> dict[str, float]:
    return _floats(_finite_blob("skohorod_metric", bench_skohorod_metric(seed)))


def bench_prohorov_thm2_family(
    seed: int = _SEED + 15104,
) -> dict[str, float]:
    return _floats(_finite_blob("prohorov_thm2", bench_prohorov_thm2(seed)))


def bench_tightness_check_family(
    seed: int = _SEED + 15105,
) -> dict[str, float]:
    return _floats(_finite_blob("tightness_check", bench_tightness_check(seed)))
