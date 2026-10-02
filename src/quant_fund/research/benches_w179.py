"""Wave-126 adapters: exec-summary survival-DL canon — deepsurv,
deephit, cox_time, nnet_surv, drsa_surv, pchazard —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cox_time import bench_cox_time
from quant_fund.models.deephit import bench_deephit
from quant_fund.models.deepsurv import bench_deepsurv
from quant_fund.models.drsa_surv import bench_drsa_surv
from quant_fund.models.nnet_surv import bench_nnet_surv
from quant_fund.models.pchazard import bench_pchazard

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


def bench_deepsurv_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("deepsurv", bench_deepsurv(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deepsurv bench failed: {exc}") from exc


def bench_deephit_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("deephit", bench_deephit(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deephit bench failed: {exc}") from exc


def bench_cox_time_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cox_time", bench_cox_time(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cox_time bench failed: {exc}") from exc


def bench_nnet_surv_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nnet_surv", bench_nnet_surv(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nnet_surv bench failed: {exc}") from exc


def bench_drsa_surv_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("drsa_surv", bench_drsa_surv(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"drsa_surv bench failed: {exc}") from exc


def bench_pchazard_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pchazard", bench_pchazard(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pchazard bench failed: {exc}") from exc
