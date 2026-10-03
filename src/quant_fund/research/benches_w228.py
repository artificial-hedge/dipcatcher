"""Wave-228 adapters: CRDT canon — gcounter, pncounter, orset,
lww_map, twopset, rga_sequence — benched on SYNTHETIC replica states.
Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gcounter import bench_gcounter
from quant_fund.models.lww_map import bench_lww_map
from quant_fund.models.orset import bench_orset
from quant_fund.models.pncounter import bench_pncounter
from quant_fund.models.rga_sequence import bench_rga_sequence
from quant_fund.models.twopset import bench_twopset

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


def bench_gcounter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gcounter", bench_gcounter(seed=_SEED + 1050)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gcounter bench failed: {exc}") from exc


def bench_lww_map_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lww_map", bench_lww_map(seed=_SEED + 1051)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lww_map bench failed: {exc}") from exc


def bench_orset_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("orset", bench_orset(seed=_SEED + 1052)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"orset bench failed: {exc}") from exc


def bench_pncounter_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pncounter", bench_pncounter(seed=_SEED + 1053)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pncounter bench failed: {exc}") from exc


def bench_rga_sequence_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rga_sequence", bench_rga_sequence(seed=_SEED + 1054)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rga_sequence bench failed: {exc}") from exc


def bench_twopset_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("twopset", bench_twopset(seed=_SEED + 1055)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"twopset bench failed: {exc}") from exc
