"""Wave-428 deformation-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.deformation_functor import (
    bench_deformation_functor,
)
from quant_fund.models.maurer_cartan import bench_maurer_cartan
from quant_fund.models.obstruction_theory import (
    bench_obstruction_theory,
)
from quant_fund.models.schlessinger import bench_schlessinger
from quant_fund.models.tangent_space_def import (
    bench_tangent_space_def,
)
from quant_fund.models.versal_deformation import (
    bench_versal_deformation,
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


def bench_deformation_functor_family(
    seed: int = _SEED + 2474,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "deformation_functor",
            bench_deformation_functor(seed),
        )
    )


def bench_schlessinger_family(
    seed: int = _SEED + 2475,
) -> dict[str, float]:
    return _floats(_finite_blob("schlessinger", bench_schlessinger(seed)))


def bench_tangent_space_def_family(
    seed: int = _SEED + 2476,
) -> dict[str, float]:
    return _floats(_finite_blob("tangent_space_def", bench_tangent_space_def(seed)))


def bench_obstruction_theory_family(
    seed: int = _SEED + 2477,
) -> dict[str, float]:
    return _floats(_finite_blob("obstruction_theory", bench_obstruction_theory(seed)))


def bench_versal_deformation_family(
    seed: int = _SEED + 2478,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "versal_deformation",
            bench_versal_deformation(seed),
        )
    )


def bench_maurer_cartan_family(
    seed: int = _SEED + 2479,
) -> dict[str, float]:
    return _floats(_finite_blob("maurer_cartan", bench_maurer_cartan(seed)))
