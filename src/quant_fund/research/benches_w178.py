"""Wave-126 adapters: exec-summary multi-task-gradient canon — pcgrad,
mgda_mtl, cagrad_mtl, gradnorm_bal, nash_mtl, imtl_g —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cagrad_mtl import bench_cagrad_mtl
from quant_fund.models.gradnorm_bal import bench_gradnorm_bal
from quant_fund.models.imtl_g import bench_imtl_g
from quant_fund.models.mgda_mtl import bench_mgda_mtl
from quant_fund.models.nash_mtl import bench_nash_mtl
from quant_fund.models.pcgrad import bench_pcgrad

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


def bench_pcgrad_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pcgrad", bench_pcgrad(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pcgrad bench failed: {exc}") from exc


def bench_mgda_mtl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mgda_mtl", bench_mgda_mtl(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mgda_mtl bench failed: {exc}") from exc


def bench_cagrad_mtl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cagrad_mtl", bench_cagrad_mtl(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cagrad_mtl bench failed: {exc}") from exc


def bench_gradnorm_bal_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gradnorm_bal", bench_gradnorm_bal(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gradnorm_bal bench failed: {exc}") from exc


def bench_nash_mtl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nash_mtl", bench_nash_mtl(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nash_mtl bench failed: {exc}") from exc


def bench_imtl_g_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("imtl_g", bench_imtl_g(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"imtl_g bench failed: {exc}") from exc
