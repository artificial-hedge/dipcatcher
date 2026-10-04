"""Wave-601 derived-geometry-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_loop import bench_derived_loop
from quant_fund.models.derived_tangent import (
    bench_derived_tangent,
)
from quant_fund.models.dg_algebra import bench_dg_algebra
from quant_fund.models.e_infinity_ring import (
    bench_e_infinity_ring,
)
from quant_fund.models.structured_space import (
    bench_structured_space,
)
from quant_fund.models.virtual_fund import bench_virtual_fund

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


def bench_dg_algebra_family(
    seed: int = _SEED + 3512,
) -> dict[str, float]:
    return _floats(_finite_blob("dg_algebra", bench_dg_algebra(seed)))


def bench_derived_loop_family(
    seed: int = _SEED + 3513,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_loop",
            bench_derived_loop(seed),
        )
    )


def bench_derived_tangent_family(
    seed: int = _SEED + 3514,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_tangent",
            bench_derived_tangent(seed),
        )
    )


def bench_virtual_fund_family(
    seed: int = _SEED + 3515,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "virtual_fund",
            bench_virtual_fund(seed),
        )
    )


def bench_structured_space_family(
    seed: int = _SEED + 3516,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "structured_space",
            bench_structured_space(seed),
        )
    )


def bench_e_infinity_ring_family(
    seed: int = _SEED + 3517,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "e_infinity_ring",
            bench_e_infinity_ring(seed),
        )
    )
