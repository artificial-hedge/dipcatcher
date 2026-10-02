"""Wave-114 adapters: GNSS & inertial canon — Gold-code PRN
generation/acquisition, Klobuchar ionospheric delay, overlapping
Allan deviation, strapdown mechanization, LAMBDA integer least
squares, and RTK double-difference positioning — each benched on
SYNTHETIC signals. Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.allan_variance import bench_allan_variance
from quant_fund.models.gold_code import bench_gold_code
from quant_fund.models.klobuchar import bench_klobuchar
from quant_fund.models.lambda_method import bench_lambda_method
from quant_fund.models.rtk import bench_rtk
from quant_fund.models.strapdown import bench_strapdown

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
                flat[f"{k}_{i}"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_gold_code_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gold_code", bench_gold_code(seed=_SEED + 672)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gold_code bench failed: {exc}") from exc


def bench_klobuchar_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("klobuchar", bench_klobuchar(seed=_SEED + 673)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"klobuchar bench failed: {exc}") from exc


def bench_allan_variance_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("allan_variance", bench_allan_variance(seed=_SEED + 674)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"allan_variance bench failed: {exc}") from exc


def bench_strapdown_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("strapdown", bench_strapdown(seed=_SEED + 675)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"strapdown bench failed: {exc}") from exc


def bench_lambda_method_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lambda_method", bench_lambda_method(seed=_SEED + 676)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lambda_method bench failed: {exc}") from exc


def bench_rtk_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rtk", bench_rtk(seed=_SEED + 677)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rtk bench failed: {exc}") from exc
