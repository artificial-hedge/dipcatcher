"""Wave-103 adapters: numerical canon III — Lanczos
tridiagonalization + Ritz values, Arnoldi/GMRES(m) for
nonsymmetric systems, HMT randomized SVD, Nyström PSD
approximation, CUR leverage-score decomposition, and
interpolative decomposition via column-pivoted QR.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arnoldi_gmres import bench_arnoldi_gmres
from quant_fund.models.cur_decomp import bench_cur_decomp
from quant_fund.models.interpolative_decomp import bench_interpolative_decomp
from quant_fund.models.lanczos import bench_lanczos
from quant_fund.models.nystrom import bench_nystrom
from quant_fund.models.randomized_svd import bench_randomized_svd

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


def bench_lanczos_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("lanczos", bench_lanczos(seed=_SEED + 606)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lanczos bench failed: {exc}") from exc


def bench_arnoldi_gmres_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("arnoldi_gmres", bench_arnoldi_gmres(seed=_SEED + 607))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"arnoldi_gmres bench failed: {exc}") from exc


def bench_randomized_svd_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("randomized_svd", bench_randomized_svd(seed=_SEED + 608))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"randomized_svd bench failed: {exc}") from exc


def bench_nystrom_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("nystrom", bench_nystrom(seed=_SEED + 609)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nystrom bench failed: {exc}") from exc


def bench_cur_decomp_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("cur_decomp", bench_cur_decomp(seed=_SEED + 610)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cur_decomp bench failed: {exc}") from exc


def bench_interpolative_decomp_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob(
                "interpolative_decomp",
                bench_interpolative_decomp(seed=_SEED + 611),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"interpolative_decomp bench failed: {exc}") from exc
