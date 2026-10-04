"""Wave-794 stochastic-control bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dynamic_programming import (
    bench_dynamic_programming,
)
from quant_fund.models.hamilton_jacobi import (
    bench_hamilton_jacobi,
)
from quant_fund.models.impulsive_control import (
    bench_impulsive_control,
)
from quant_fund.models.quasi_variational import (
    bench_quasi_variational,
)
from quant_fund.models.verification_thm import (
    bench_verification_thm,
)
from quant_fund.models.viscosity_solution import (
    bench_viscosity_solution,
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


def bench_dynamic_programming_family(
    seed: int = _SEED + 18300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dynamic_programming",
            bench_dynamic_programming(seed),
        )
    )


def bench_verification_thm_family(
    seed: int = _SEED + 18301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "verification_thm",
            bench_verification_thm(seed),
        )
    )


def bench_hamilton_jacobi_family(
    seed: int = _SEED + 18302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hamilton_jacobi",
            bench_hamilton_jacobi(seed),
        )
    )


def bench_viscosity_solution_family(
    seed: int = _SEED + 18303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "viscosity_solution",
            bench_viscosity_solution(seed),
        )
    )


def bench_quasi_variational_family(
    seed: int = _SEED + 18304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quasi_variational",
            bench_quasi_variational(seed),
        )
    )


def bench_impulsive_control_family(
    seed: int = _SEED + 18305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "impulsive_control",
            bench_impulsive_control(seed),
        )
    )
