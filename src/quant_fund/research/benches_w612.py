"""Wave-612 commutative-algebra-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.discrete_valuation import (
    bench_discrete_valuation,
)
from quant_fund.models.factorial_ring import (
    bench_factorial_ring,
)
from quant_fund.models.gorenstein_ring import (
    bench_gorenstein_ring,
)
from quant_fund.models.jacobson_ring import (
    bench_jacobson_ring,
)
from quant_fund.models.normal_ring import bench_normal_ring
from quant_fund.models.regular_ring import bench_regular_ring

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


def bench_regular_ring_family(
    seed: int = _SEED + 3578,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "regular_ring",
            bench_regular_ring(seed),
        )
    )


def bench_gorenstein_ring_family(
    seed: int = _SEED + 3579,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gorenstein_ring",
            bench_gorenstein_ring(seed),
        )
    )


def bench_normal_ring_family(
    seed: int = _SEED + 3580,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "normal_ring",
            bench_normal_ring(seed),
        )
    )


def bench_factorial_ring_family(
    seed: int = _SEED + 3581,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "factorial_ring",
            bench_factorial_ring(seed),
        )
    )


def bench_jacobson_ring_family(
    seed: int = _SEED + 3582,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "jacobson_ring",
            bench_jacobson_ring(seed),
        )
    )


def bench_discrete_valuation_family(
    seed: int = _SEED + 3583,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "discrete_valuation",
            bench_discrete_valuation(seed),
        )
    )
