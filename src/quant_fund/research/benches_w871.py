"""Wave-871 Krylov-solver bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arnoldi_eig import (
    bench_arnoldi_eig,
)
from quant_fund.models.bicg_solver import (
    bench_bicg_solver,
)
from quant_fund.models.cg_solver import (
    bench_cg_solver,
)
from quant_fund.models.gmres_solver import (
    bench_gmres_solver,
)
from quant_fund.models.lanczos_eig import (
    bench_lanczos_eig,
)
from quant_fund.models.lsqr_solver import (
    bench_lsqr_solver,
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


def bench_cg_solver_family(
    seed: int = _SEED + 25900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cg_solver",
            bench_cg_solver(seed),
        )
    )


def bench_gmres_solver_family(
    seed: int = _SEED + 25901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gmres_solver",
            bench_gmres_solver(seed),
        )
    )


def bench_bicg_solver_family(
    seed: int = _SEED + 25902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bicg_solver",
            bench_bicg_solver(seed),
        )
    )


def bench_arnoldi_eig_family(
    seed: int = _SEED + 25903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "arnoldi_eig",
            bench_arnoldi_eig(seed),
        )
    )


def bench_lanczos_eig_family(
    seed: int = _SEED + 25904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lanczos_eig",
            bench_lanczos_eig(seed),
        )
    )


def bench_lsqr_solver_family(
    seed: int = _SEED + 25905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lsqr_solver",
            bench_lsqr_solver(seed),
        )
    )
