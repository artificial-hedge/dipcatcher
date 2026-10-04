"""Wave-564 geometric-group-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.asymptotic_cone import bench_asymptotic_cone
from quant_fund.models.baumslag_solitar import bench_baumslag_solitar
from quant_fund.models.gromov_hyperbolic import (
    bench_gromov_hyperbolic,
)
from quant_fund.models.quasi_isometry import bench_quasi_isometry
from quant_fund.models.thin_triangle import bench_thin_triangle
from quant_fund.models.word_problem import bench_word_problem

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


def bench_gromov_hyperbolic_family(
    seed: int = _SEED + 3290,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gromov_hyperbolic",
            bench_gromov_hyperbolic(seed),
        )
    )


def bench_quasi_isometry_family(
    seed: int = _SEED + 3291,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quasi_isometry",
            bench_quasi_isometry(seed),
        )
    )


def bench_thin_triangle_family(seed: int = _SEED + 3292) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "thin_triangle",
            bench_thin_triangle(seed),
        )
    )


def bench_word_problem_family(seed: int = _SEED + 3293) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "word_problem",
            bench_word_problem(seed),
        )
    )


def bench_baumslag_solitar_family(
    seed: int = _SEED + 3294,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "baumslag_solitar",
            bench_baumslag_solitar(seed),
        )
    )


def bench_asymptotic_cone_family(
    seed: int = _SEED + 3295,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "asymptotic_cone",
            bench_asymptotic_cone(seed),
        )
    )
