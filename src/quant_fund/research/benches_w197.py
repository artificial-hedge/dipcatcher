"""Wave-126 adapters: exec-summary inventory-theory canon — base_stock,
eoq_model, wagner_whitin, newsvendor, ss_policy, clark_scarf —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.base_stock import bench_base_stock
from quant_fund.models.clark_scarf import bench_clark_scarf
from quant_fund.models.eoq_model import bench_eoq_model
from quant_fund.models.newsvendor import bench_newsvendor
from quant_fund.models.ss_policy import bench_ss_policy
from quant_fund.models.wagner_whitin import bench_wagner_whitin

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


def bench_base_stock_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("base_stock", bench_base_stock(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"base_stock bench failed: {exc}") from exc


def bench_eoq_model_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("eoq_model", bench_eoq_model(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"eoq_model bench failed: {exc}") from exc


def bench_wagner_whitin_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("wagner_whitin", bench_wagner_whitin(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"wagner_whitin bench failed: {exc}") from exc


def bench_newsvendor_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("newsvendor", bench_newsvendor(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"newsvendor bench failed: {exc}") from exc


def bench_ss_policy_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ss_policy", bench_ss_policy(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ss_policy bench failed: {exc}") from exc


def bench_clark_scarf_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("clark_scarf", bench_clark_scarf(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"clark_scarf bench failed: {exc}") from exc
