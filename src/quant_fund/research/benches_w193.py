"""Wave-126 adapters: exec-summary optimal-control canon — mpc_qp,
lqr_control, pmp_bangbang, ddp_solve, mppi_control, lqg_control —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ddp_solve import bench_ddp_solve
from quant_fund.models.lqg_control import bench_lqg_control
from quant_fund.models.lqr_control import bench_lqr_control
from quant_fund.models.mpc_qp import bench_mpc_qp
from quant_fund.models.mppi_control import bench_mppi_control
from quant_fund.models.pmp_bangbang import bench_pmp_bangbang

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
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_mpc_qp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mpc_qp", bench_mpc_qp(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mpc_qp bench failed: {exc}") from exc


def bench_lqr_control_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lqr_control", bench_lqr_control(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lqr_control bench failed: {exc}") from exc


def bench_pmp_bangbang_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pmp_bangbang", bench_pmp_bangbang(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pmp_bangbang bench failed: {exc}") from exc


def bench_ddp_solve_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ddp_solve", bench_ddp_solve(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ddp_solve bench failed: {exc}") from exc


def bench_mppi_control_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mppi_control", bench_mppi_control(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mppi_control bench failed: {exc}") from exc


def bench_lqg_control_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lqg_control", bench_lqg_control(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lqg_control bench failed: {exc}") from exc
