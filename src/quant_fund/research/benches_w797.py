"""Wave-797 2BSDE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.doubly_bsde import (
    bench_doubly_bsde,
)
from quant_fund.models.obstacle_bsde import (
    bench_obstacle_bsde,
)
from quant_fund.models.quadratic_bsde import (
    bench_quadratic_bsde,
)
from quant_fund.models.reflected_bsde2 import (
    bench_reflected_bsde2,
)
from quant_fund.models.second_bsde import (
    bench_second_bsde,
)
from quant_fund.models.super_linear import (
    bench_super_linear,
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


def bench_second_bsde_family(
    seed: int = _SEED + 18600,
) -> dict[str, float]:
    return _floats(_finite_blob("second_bsde", bench_second_bsde(seed)))


def bench_doubly_bsde_family(
    seed: int = _SEED + 18601,
) -> dict[str, float]:
    return _floats(_finite_blob("doubly_bsde", bench_doubly_bsde(seed)))


def bench_reflected_bsde2_family(
    seed: int = _SEED + 18602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "reflected_bsde2",
            bench_reflected_bsde2(seed),
        )
    )


def bench_obstacle_bsde_family(
    seed: int = _SEED + 18603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "obstacle_bsde",
            bench_obstacle_bsde(seed),
        )
    )


def bench_quadratic_bsde_family(
    seed: int = _SEED + 18604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quadratic_bsde",
            bench_quadratic_bsde(seed),
        )
    )


def bench_super_linear_family(
    seed: int = _SEED + 18605,
) -> dict[str, float]:
    return _floats(_finite_blob("super_linear", bench_super_linear(seed)))
