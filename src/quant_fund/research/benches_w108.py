"""Wave-108 adapters: matrix functions & matrix equations canon
— scaling-and-squaring Padé expm, Denman–Beavers matrix sqrt,
Bartels–Stewart Sylvester/Lyapunov, Hamiltonian/Kleinman CARE,
scaled-Newton matrix sign, and Levinson Toeplitz — each benched
on SYNTHETIC systems with closed-form or dense-solve references.
Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.expm_pade import bench_expm_pade
from quant_fund.models.matrix_sign import bench_matrix_sign
from quant_fund.models.matrix_sqrt import bench_matrix_sqrt
from quant_fund.models.riccati_care import bench_riccati_care
from quant_fund.models.sylvester import bench_sylvester
from quant_fund.models.toeplitz_solve import bench_toeplitz_solve

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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
                flat[f"{k}_{i}"] = f
    return flat


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_expm_pade_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("expm_pade", bench_expm_pade(seed=_SEED + 636)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"expm_pade bench failed: {exc}") from exc


def bench_matrix_sqrt_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("matrix_sqrt", bench_matrix_sqrt(seed=_SEED + 637)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"matrix_sqrt bench failed: {exc}") from exc


def bench_sylvester_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("sylvester", bench_sylvester(seed=_SEED + 638)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sylvester bench failed: {exc}") from exc


def bench_riccati_care_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("riccati_care", bench_riccati_care(seed=_SEED + 639))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"riccati_care bench failed: {exc}") from exc


def bench_matrix_sign_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("matrix_sign", bench_matrix_sign(seed=_SEED + 640)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"matrix_sign bench failed: {exc}") from exc


def bench_toeplitz_solve_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("toeplitz_solve", bench_toeplitz_solve(seed=_SEED + 641))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"toeplitz_solve bench failed: {exc}") from exc
