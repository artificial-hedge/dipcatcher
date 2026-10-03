"""Wave-217 adapters: estimation/filtering canon — hinf_filter,
cubature_kalman, mhe, variational_bayes, huber_filter, particle_smoother —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cubature_kalman import bench_cubature_kalman
from quant_fund.models.hinf_filter import bench_hinf_filter
from quant_fund.models.huber_filter import bench_huber_filter
from quant_fund.models.mhe import bench_mhe
from quant_fund.models.particle_smoother import bench_particle_smoother
from quant_fund.models.variational_bayes import bench_variational_bayes

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


def bench_huber_filter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("huber_filter", bench_huber_filter(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"huber_filter bench failed: {exc}") from exc


def bench_hinf_filter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hinf_filter", bench_hinf_filter(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hinf_filter bench failed: {exc}") from exc


def bench_variational_bayes_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("variational_bayes", bench_variational_bayes(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"variational_bayes bench failed: {exc}") from exc


def bench_cubature_kalman_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cubature_kalman", bench_cubature_kalman(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cubature_kalman bench failed: {exc}") from exc


def bench_mhe_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mhe", bench_mhe(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mhe bench failed: {exc}") from exc


def bench_particle_smoother_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("particle_smoother", bench_particle_smoother(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"particle_smoother bench failed: {exc}") from exc
