"""Wave-897 ODE-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dahlquist_test import (
    bench_dahlquist_test,
)
from quant_fund.models.explicit_midpoint import (
    bench_explicit_midpoint,
)
from quant_fund.models.heun_method import (
    bench_heun_method,
)
from quant_fund.models.linear_multistep import (
    bench_linear_multistep,
)
from quant_fund.models.order_barrier import (
    bench_order_barrier,
)
from quant_fund.models.trapezoid_rule import (
    bench_trapezoid_rule,
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


def bench_linear_multistep_family(
    seed: int = _SEED + 28500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "linear_multistep",
            bench_linear_multistep(seed),
        )
    )


def bench_dahlquist_test_family(
    seed: int = _SEED + 28501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dahlquist_test",
            bench_dahlquist_test(seed),
        )
    )


def bench_explicit_midpoint_family(
    seed: int = _SEED + 28502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "explicit_midpoint",
            bench_explicit_midpoint(seed),
        )
    )


def bench_heun_method_family(
    seed: int = _SEED + 28503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "heun_method",
            bench_heun_method(seed),
        )
    )


def bench_trapezoid_rule_family(
    seed: int = _SEED + 28504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "trapezoid_rule",
            bench_trapezoid_rule(seed),
        )
    )


def bench_order_barrier_family(
    seed: int = _SEED + 28505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "order_barrier",
            bench_order_barrier(seed),
        )
    )
