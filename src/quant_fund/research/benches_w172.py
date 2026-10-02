"""Wave-126 adapters: exec-summary bandit-exotics canon — psrl,
gittins_index, whittle_restless, cucb, corrupt_bandit, neural_ucb —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.corrupt_bandit import bench_corrupt_bandit
from quant_fund.models.cucb import bench_cucb
from quant_fund.models.gittins_index import bench_gittins_index
from quant_fund.models.neural_ucb import bench_neural_ucb
from quant_fund.models.psrl import bench_psrl
from quant_fund.models.whittle_restless import bench_whittle_restless

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


def bench_psrl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("psrl", bench_psrl(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"psrl bench failed: {exc}") from exc


def bench_gittins_index_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gittins_index", bench_gittins_index(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gittins_index bench failed: {exc}") from exc


def bench_whittle_restless_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("whittle_restless", bench_whittle_restless(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"whittle_restless bench failed: {exc}") from exc


def bench_cucb_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cucb", bench_cucb(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cucb bench failed: {exc}") from exc


def bench_corrupt_bandit_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("corrupt_bandit", bench_corrupt_bandit(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"corrupt_bandit bench failed: {exc}") from exc


def bench_neural_ucb_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("neural_ucb", bench_neural_ucb(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"neural_ucb bench failed: {exc}") from exc
