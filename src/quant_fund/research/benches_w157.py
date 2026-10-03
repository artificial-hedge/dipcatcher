"""Wave-126 adapters: exec-summary anomaly-detection canon — deep_svdd,
dagmm, usad, anom_transformer, rrcf, tranad —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anom_transformer import bench_anom_transformer
from quant_fund.models.dagmm import bench_dagmm
from quant_fund.models.deep_svdd import bench_deep_svdd
from quant_fund.models.rrcf import bench_rrcf
from quant_fund.models.tranad import bench_tranad
from quant_fund.models.usad import bench_usad

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


def bench_deep_svdd_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("deep_svdd", bench_deep_svdd(seed=_SEED + 930)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deep_svdd bench failed: {exc}") from exc


def bench_dagmm_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dagmm", bench_dagmm(seed=_SEED + 931)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dagmm bench failed: {exc}") from exc


def bench_usad_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("usad", bench_usad(seed=_SEED + 932)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"usad bench failed: {exc}") from exc


def bench_anom_transformer_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("anom_transformer", bench_anom_transformer(seed=_SEED + 933)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"anom_transformer bench failed: {exc}") from exc


def bench_rrcf_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rrcf", bench_rrcf(seed=_SEED + 934)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rrcf bench failed: {exc}") from exc


def bench_tranad_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tranad", bench_tranad(seed=_SEED + 935)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tranad bench failed: {exc}") from exc
