"""Wave-126 adapters: exec-summary compression canon — magnitude_pruning,
lottery_ticket, quant_int8, kd_distill, lowrank_factor, fisher_prune —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.fisher_prune import bench_fisher_prune
from quant_fund.models.kd_distill import bench_kd_distill
from quant_fund.models.lottery_ticket import bench_lottery_ticket
from quant_fund.models.lowrank_factor import bench_lowrank_factor
from quant_fund.models.magnitude_pruning import bench_magnitude_pruning
from quant_fund.models.quant_int8 import bench_quant_int8

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


def bench_magnitude_pruning_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("magnitude_pruning", bench_magnitude_pruning(seed=_SEED + 888)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"magnitude_pruning bench failed: {exc}") from exc


def bench_lottery_ticket_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lottery_ticket", bench_lottery_ticket(seed=_SEED + 889)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lottery_ticket bench failed: {exc}") from exc


def bench_quant_int8_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("quant_int8", bench_quant_int8(seed=_SEED + 890)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"quant_int8 bench failed: {exc}") from exc


def bench_kd_distill_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("kd_distill", bench_kd_distill(seed=_SEED + 891)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kd_distill bench failed: {exc}") from exc


def bench_lowrank_factor_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lowrank_factor", bench_lowrank_factor(seed=_SEED + 892)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lowrank_factor bench failed: {exc}") from exc


def bench_fisher_prune_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fisher_prune", bench_fisher_prune(seed=_SEED + 893)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fisher_prune bench failed: {exc}") from exc
