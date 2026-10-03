"""Wave-100 adapters: numerical canon II — homotopy
continuation + Euler-Newton predictor-corrector, Anderson
acceleration for fixed points, Aitken/Shanks/Wynn/Richardson
sequence acceleration, CGLS + LSQR iterative least squares,
low-discrepancy QMC sequences, and Stoermer-Verlet/Yoshida
symplectic integrators.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anderson_accel import bench_anderson
from quant_fund.models.homotopy_continuation import bench_homotopy
from quant_fund.models.iterative_ls import bench_iterative_ls
from quant_fund.models.qmc_sequences import bench_qmc
from quant_fund.models.sequence_accel import bench_sequence_accel
from quant_fund.models.symplectic_ode import bench_symplectic

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


def bench_homotopy_continuation_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("homotopy_continuation", bench_homotopy(seed=_SEED + 588))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"homotopy_continuation bench failed: {exc}") from exc


def bench_anderson_accel_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("anderson_accel", bench_anderson(seed=_SEED + 589)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"anderson_accel bench failed: {exc}") from exc


def bench_sequence_accel_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("sequence_accel", bench_sequence_accel(seed=_SEED + 590))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sequence_accel bench failed: {exc}") from exc


def bench_iterative_ls_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("iterative_ls", bench_iterative_ls(seed=_SEED + 591))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"iterative_ls bench failed: {exc}") from exc


def bench_qmc_sequences_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("qmc_sequences", bench_qmc(seed=_SEED + 592)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qmc_sequences bench failed: {exc}") from exc


def bench_symplectic_ode_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("symplectic_ode", bench_symplectic(seed=_SEED + 593))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"symplectic_ode bench failed: {exc}") from exc
