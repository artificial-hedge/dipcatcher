"""Wave-126 adapters: exec-summary privacy canon — dp_sgd,
secure_agg, fedavg_hetero, pate_teacher, gradient_leakage, canary_exposure —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.canary_exposure import bench_canary_exposure
from quant_fund.models.dp_sgd import bench_dp_sgd
from quant_fund.models.fedavg_hetero import bench_fedavg_hetero
from quant_fund.models.gradient_leakage import bench_gradient_leakage
from quant_fund.models.pate_teacher import bench_pate_teacher
from quant_fund.models.secure_agg import bench_secure_agg

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


def bench_dp_sgd_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dp_sgd", bench_dp_sgd(seed=_SEED + 900)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dp_sgd bench failed: {exc}") from exc


def bench_secure_agg_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("secure_agg", bench_secure_agg(seed=_SEED + 901)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"secure_agg bench failed: {exc}") from exc


def bench_fedavg_hetero_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fedavg_hetero", bench_fedavg_hetero(seed=_SEED + 902)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fedavg_hetero bench failed: {exc}") from exc


def bench_pate_teacher_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pate_teacher", bench_pate_teacher(seed=_SEED + 903)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pate_teacher bench failed: {exc}") from exc


def bench_gradient_leakage_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gradient_leakage", bench_gradient_leakage(seed=_SEED + 904)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gradient_leakage bench failed: {exc}") from exc


def bench_canary_exposure_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("canary_exposure", bench_canary_exposure(seed=_SEED + 905)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"canary_exposure bench failed: {exc}") from exc
