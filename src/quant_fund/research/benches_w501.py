"""Wave-501 F-singularity bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.f_pure import bench_f_pure
from quant_fund.models.f_rational import bench_f_rational
from quant_fund.models.f_regular import bench_f_regular
from quant_fund.models.f_threshold import bench_f_threshold
from quant_fund.models.test_ideal import bench_test_ideal
from quant_fund.models.tight_closure import bench_tight_closure

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


def bench_f_regular_family(seed: int = _SEED + 2912) -> dict[str, float]:
    return _floats(_finite_blob("f_regular", bench_f_regular(seed)))


def bench_f_rational_family(seed: int = _SEED + 2913) -> dict[str, float]:
    return _floats(_finite_blob("f_rational", bench_f_rational(seed)))


def bench_f_pure_family(seed: int = _SEED + 2914) -> dict[str, float]:
    return _floats(_finite_blob("f_pure", bench_f_pure(seed)))


def bench_f_threshold_family(seed: int = _SEED + 2915) -> dict[str, float]:
    return _floats(_finite_blob("f_threshold", bench_f_threshold(seed)))


def bench_test_ideal_family(seed: int = _SEED + 2916) -> dict[str, float]:
    return _floats(_finite_blob("test_ideal", bench_test_ideal(seed)))


def bench_tight_closure_family(seed: int = _SEED + 2917) -> dict[str, float]:
    return _floats(_finite_blob("tight_closure", bench_tight_closure(seed)))
