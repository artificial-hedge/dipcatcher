"""Wave-214 adapters: online-algorithms canon — ski_rental,
marking_paging, work_function_kserver, ranking_matching, secretary_prophet, online_gradient —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.marking_paging import bench_marking_paging
from quant_fund.models.online_gradient import bench_online_gradient
from quant_fund.models.ranking_matching import bench_ranking_matching
from quant_fund.models.secretary_prophet import bench_secretary_prophet
from quant_fund.models.ski_rental import bench_ski_rental
from quant_fund.models.work_function_kserver import bench_work_function_kserver

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


def bench_secretary_prophet_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("secretary_prophet", bench_secretary_prophet(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"secretary_prophet bench failed: {exc}") from exc


def bench_ski_rental_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ski_rental", bench_ski_rental(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ski_rental bench failed: {exc}") from exc


def bench_ranking_matching_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ranking_matching", bench_ranking_matching(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ranking_matching bench failed: {exc}") from exc


def bench_marking_paging_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("marking_paging", bench_marking_paging(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"marking_paging bench failed: {exc}") from exc


def bench_work_function_kserver_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("work_function_kserver", bench_work_function_kserver(seed=_SEED + 964))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"work_function_kserver bench failed: {exc}") from exc


def bench_online_gradient_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("online_gradient", bench_online_gradient(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"online_gradient bench failed: {exc}") from exc
