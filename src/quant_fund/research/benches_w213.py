"""Wave-213 adapters: SDP/relaxation canon — sdp_maxcut,
eigenvalue_opt, sos_certificate, qcqp_relax, spectral_bisection, hoffman_bound —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.eigenvalue_opt import bench_eigenvalue_opt
from quant_fund.models.hoffman_bound import bench_hoffman_bound
from quant_fund.models.qcqp_relax import bench_qcqp_relax
from quant_fund.models.sdp_maxcut import bench_sdp_maxcut
from quant_fund.models.sos_certificate import bench_sos_certificate
from quant_fund.models.spectral_bisection import bench_spectral_bisection

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


def bench_spectral_bisection_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("spectral_bisection", bench_spectral_bisection(seed=_SEED + 960))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"spectral_bisection bench failed: {exc}") from exc


def bench_sdp_maxcut_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sdp_maxcut", bench_sdp_maxcut(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sdp_maxcut bench failed: {exc}") from exc


def bench_qcqp_relax_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qcqp_relax", bench_qcqp_relax(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qcqp_relax bench failed: {exc}") from exc


def bench_eigenvalue_opt_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("eigenvalue_opt", bench_eigenvalue_opt(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"eigenvalue_opt bench failed: {exc}") from exc


def bench_sos_certificate_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sos_certificate", bench_sos_certificate(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sos_certificate bench failed: {exc}") from exc


def bench_hoffman_bound_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hoffman_bound", bench_hoffman_bound(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hoffman_bound bench failed: {exc}") from exc
