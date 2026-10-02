"""Wave-126 adapters: exec-summary TS-foundation canon — chronos_lite,
timesfm_lite, moirai_lite, lagllama_lite, timer_lite, moment_lite —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chronos_lite import bench_chronos_lite
from quant_fund.models.lagllama_lite import bench_lagllama_lite
from quant_fund.models.moirai_lite import bench_moirai_lite
from quant_fund.models.moment_lite import bench_moment_lite
from quant_fund.models.timer_lite import bench_timer_lite
from quant_fund.models.timesfm_lite import bench_timesfm_lite

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


def bench_chronos_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("chronos_lite", bench_chronos_lite(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"chronos_lite bench failed: {exc}") from exc


def bench_timesfm_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("timesfm_lite", bench_timesfm_lite(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"timesfm_lite bench failed: {exc}") from exc


def bench_moirai_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("moirai_lite", bench_moirai_lite(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"moirai_lite bench failed: {exc}") from exc


def bench_lagllama_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lagllama_lite", bench_lagllama_lite(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lagllama_lite bench failed: {exc}") from exc


def bench_timer_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("timer_lite", bench_timer_lite(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"timer_lite bench failed: {exc}") from exc


def bench_moment_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("moment_lite", bench_moment_lite(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"moment_lite bench failed: {exc}") from exc
