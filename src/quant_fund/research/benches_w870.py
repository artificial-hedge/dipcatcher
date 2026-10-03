"""Wave-870 optimization bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.augmented_lagrangian import (
    bench_augmented_lagrangian,
)
from quant_fund.models.conjugate_opt import (
    bench_conjugate_opt,
)
from quant_fund.models.grad_descent_nest import (
    bench_grad_descent_nest,
)
from quant_fund.models.interior_point2 import (
    bench_interior_point2,
)
from quant_fund.models.newton_method import (
    bench_newton_method,
)
from quant_fund.models.quasi_newton_lbfgs import (
    bench_quasi_newton_lbfgs,
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


def bench_newton_method_family(
    seed: int = _SEED + 25800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "newton_method",
            bench_newton_method(seed),
        )
    )


def bench_quasi_newton_lbfgs_family(
    seed: int = _SEED + 25801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quasi_newton_lbfgs",
            bench_quasi_newton_lbfgs(seed),
        )
    )


def bench_augmented_lagrangian_family(
    seed: int = _SEED + 25802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "augmented_lagrangian",
            bench_augmented_lagrangian(seed),
        )
    )


def bench_interior_point2_family(
    seed: int = _SEED + 25803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "interior_point2",
            bench_interior_point2(seed),
        )
    )


def bench_grad_descent_nest_family(
    seed: int = _SEED + 25804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "grad_descent_nest",
            bench_grad_descent_nest(seed),
        )
    )


def bench_conjugate_opt_family(
    seed: int = _SEED + 25805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "conjugate_opt",
            bench_conjugate_opt(seed),
        )
    )
