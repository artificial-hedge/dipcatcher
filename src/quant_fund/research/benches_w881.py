"""Wave-881 Krylov-solver bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bicgstab2 import (
    bench_bicgstab2,
)
from quant_fund.models.block_cg import (
    bench_block_cg,
)
from quant_fund.models.cgs_solver import (
    bench_cgs_solver,
)
from quant_fund.models.minres_solver import (
    bench_minres_solver,
)
from quant_fund.models.qmr_solver import (
    bench_qmr_solver,
)
from quant_fund.models.tfqmr import (
    bench_tfqmr,
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


def bench_minres_solver_family(
    seed: int = _SEED + 26900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "minres_solver",
            bench_minres_solver(seed),
        )
    )


def bench_cgs_solver_family(
    seed: int = _SEED + 26901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cgs_solver",
            bench_cgs_solver(seed),
        )
    )


def bench_tfqmr_family(
    seed: int = _SEED + 26902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tfqmr",
            bench_tfqmr(seed),
        )
    )


def bench_qmr_solver_family(
    seed: int = _SEED + 26903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "qmr_solver",
            bench_qmr_solver(seed),
        )
    )


def bench_bicgstab2_family(
    seed: int = _SEED + 26904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bicgstab2",
            bench_bicgstab2(seed),
        )
    )


def bench_block_cg_family(
    seed: int = _SEED + 26905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "block_cg",
            bench_block_cg(seed),
        )
    )
