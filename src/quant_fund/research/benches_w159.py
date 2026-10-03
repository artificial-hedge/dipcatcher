"""Wave-126 adapters: exec-summary optimizer canon — muon_opt,
lion_opt, sophia_opt, lookahead_opt, lamb_opt, adafactor_opt —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adafactor_opt import bench_adafactor_opt
from quant_fund.models.lamb_opt import bench_lamb_opt
from quant_fund.models.lion_opt import bench_lion_opt
from quant_fund.models.lookahead_opt import bench_lookahead_opt
from quant_fund.models.muon_opt import bench_muon_opt
from quant_fund.models.sophia_opt import bench_sophia_opt

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


def bench_muon_opt_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("muon_opt", bench_muon_opt(seed=_SEED + 942)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"muon_opt bench failed: {exc}") from exc


def bench_lion_opt_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lion_opt", bench_lion_opt(seed=_SEED + 943)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lion_opt bench failed: {exc}") from exc


def bench_sophia_opt_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sophia_opt", bench_sophia_opt(seed=_SEED + 944)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sophia_opt bench failed: {exc}") from exc


def bench_lookahead_opt_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lookahead_opt", bench_lookahead_opt(seed=_SEED + 945)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lookahead_opt bench failed: {exc}") from exc


def bench_lamb_opt_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lamb_opt", bench_lamb_opt(seed=_SEED + 946)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lamb_opt bench failed: {exc}") from exc


def bench_adafactor_opt_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("adafactor_opt", bench_adafactor_opt(seed=_SEED + 947)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"adafactor_opt bench failed: {exc}") from exc
