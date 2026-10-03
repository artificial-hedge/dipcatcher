"""Wave-883 fixed-point-acceleration bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cv_optimal import (
    bench_cv_optimal,
)
from quant_fund.models.is_drift import (
    bench_is_drift,
)
from quant_fund.models.min_var_closure import (
    bench_min_var_closure,
)
from quant_fund.models.nest_accel import (
    bench_nest_accel,
)
from quant_fund.models.subgradient_descent import (
    bench_subgradient_descent,
)
from quant_fund.models.tangent_predictor import (
    bench_tangent_predictor,
)

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


def bench_tangent_predictor_family(
    seed: int = _SEED + 27100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tangent_predictor",
            bench_tangent_predictor(seed),
        )
    )


def bench_is_drift_family(
    seed: int = _SEED + 27101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "is_drift",
            bench_is_drift(seed),
        )
    )


def bench_cv_optimal_family(
    seed: int = _SEED + 27102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cv_optimal",
            bench_cv_optimal(seed),
        )
    )


def bench_nest_accel_family(
    seed: int = _SEED + 27103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nest_accel",
            bench_nest_accel(seed),
        )
    )


def bench_subgradient_descent_family(
    seed: int = _SEED + 27104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "subgradient_descent",
            bench_subgradient_descent(seed),
        )
    )


def bench_min_var_closure_family(
    seed: int = _SEED + 27105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "min_var_closure",
            bench_min_var_closure(seed),
        )
    )
