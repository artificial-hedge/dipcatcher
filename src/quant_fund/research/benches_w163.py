"""Wave-126 adapters: exec-summary lifelong-CL canon — packnet_cl,
lwf_cl, der_cl, agem_cl, piggyback_cl, hat_cl —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.agem_cl import bench_agem_cl
from quant_fund.models.der_cl import bench_der_cl
from quant_fund.models.hat_cl import bench_hat_cl
from quant_fund.models.lwf_cl import bench_lwf_cl
from quant_fund.models.packnet_cl import bench_packnet_cl
from quant_fund.models.piggyback_cl import bench_piggyback_cl

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


def bench_packnet_cl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("packnet_cl", bench_packnet_cl(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"packnet_cl bench failed: {exc}") from exc


def bench_lwf_cl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lwf_cl", bench_lwf_cl(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lwf_cl bench failed: {exc}") from exc


def bench_der_cl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("der_cl", bench_der_cl(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"der_cl bench failed: {exc}") from exc


def bench_agem_cl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("agem_cl", bench_agem_cl(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"agem_cl bench failed: {exc}") from exc


def bench_piggyback_cl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("piggyback_cl", bench_piggyback_cl(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"piggyback_cl bench failed: {exc}") from exc


def bench_hat_cl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hat_cl", bench_hat_cl(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hat_cl bench failed: {exc}") from exc
