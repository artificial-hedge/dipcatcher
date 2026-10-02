"""Wave-209 adapters: reliability-engineering canon — weibull_life,
fault_tree, ram_markov, fmea_rpn, life_stress, redundancy_block —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.fault_tree import bench_fault_tree
from quant_fund.models.fmea_rpn import bench_fmea_rpn
from quant_fund.models.life_stress import bench_life_stress
from quant_fund.models.ram_markov import bench_ram_markov
from quant_fund.models.redundancy_block import bench_redundancy_block
from quant_fund.models.weibull_life import bench_weibull_life

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


def bench_life_stress_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("life_stress", bench_life_stress(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"life_stress bench failed: {exc}") from exc


def bench_weibull_life_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("weibull_life", bench_weibull_life(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"weibull_life bench failed: {exc}") from exc


def bench_fmea_rpn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fmea_rpn", bench_fmea_rpn(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fmea_rpn bench failed: {exc}") from exc


def bench_fault_tree_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fault_tree", bench_fault_tree(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fault_tree bench failed: {exc}") from exc


def bench_ram_markov_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ram_markov", bench_ram_markov(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ram_markov bench failed: {exc}") from exc


def bench_redundancy_block_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("redundancy_block", bench_redundancy_block(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"redundancy_block bench failed: {exc}") from exc
