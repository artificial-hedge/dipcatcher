"""Wave-126 adapters: exec-summary eigen-decomposition canon — hessenberg_red,
power_iter, qr_eig, inverse_iter, jacobi_eig, bidiag_svd —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bidiag_svd import bench_bidiag_svd
from quant_fund.models.hessenberg_red import bench_hessenberg_red
from quant_fund.models.inverse_iter import bench_inverse_iter
from quant_fund.models.jacobi_eig import bench_jacobi_eig
from quant_fund.models.power_iter import bench_power_iter
from quant_fund.models.qr_eig import bench_qr_eig

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


def bench_hessenberg_red_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hessenberg_red", bench_hessenberg_red(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hessenberg_red bench failed: {exc}") from exc


def bench_power_iter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("power_iter", bench_power_iter(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"power_iter bench failed: {exc}") from exc


def bench_qr_eig_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qr_eig", bench_qr_eig(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qr_eig bench failed: {exc}") from exc


def bench_inverse_iter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("inverse_iter", bench_inverse_iter(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"inverse_iter bench failed: {exc}") from exc


def bench_jacobi_eig_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("jacobi_eig", bench_jacobi_eig(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"jacobi_eig bench failed: {exc}") from exc


def bench_bidiag_svd_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bidiag_svd", bench_bidiag_svd(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bidiag_svd bench failed: {exc}") from exc
