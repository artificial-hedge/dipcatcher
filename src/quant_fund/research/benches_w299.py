"""Wave-299 game-playing-2 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.expectimax import bench_expectimax
from quant_fund.models.isomcts import bench_isomcts
from quant_fund.models.mast_playout import bench_mast_playout
from quant_fund.models.rave_mc import bench_rave_mc
from quant_fund.models.retrograde_wdl import bench_retrograde_wdl
from quant_fund.models.tablebase_dtm import bench_tablebase_dtm

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


def bench_tablebase_dtm_family(seed: int = _SEED + 1700) -> dict[str, float]:
    return _floats(_finite_blob("tablebase_dtm", bench_tablebase_dtm(seed)))


def bench_retrograde_wdl_family(seed: int = _SEED + 1701) -> dict[str, float]:
    return _floats(_finite_blob("retrograde_wdl", bench_retrograde_wdl(seed)))


def bench_rave_mc_family(seed: int = _SEED + 1702) -> dict[str, float]:
    return _floats(_finite_blob("rave_mc", bench_rave_mc(seed)))


def bench_mast_playout_family(seed: int = _SEED + 1703) -> dict[str, float]:
    return _floats(_finite_blob("mast_playout", bench_mast_playout(seed)))


def bench_expectimax_family(seed: int = _SEED + 1704) -> dict[str, float]:
    return _floats(_finite_blob("expectimax", bench_expectimax(seed)))


def bench_isomcts_family(seed: int = _SEED + 1705) -> dict[str, float]:
    return _floats(_finite_blob("isomcts", bench_isomcts(seed)))
