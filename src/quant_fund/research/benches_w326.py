"""Wave-326 verification-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bisim_refine import bench_bisim_refine
from quant_fund.models.ctl_mc import bench_ctl_mc
from quant_fund.models.nba_emptiness import bench_nba_emptiness
from quant_fund.models.parity_game import bench_parity_game
from quant_fund.models.timed_automata import bench_timed_automata
from quant_fund.models.wsts_cover import bench_wsts_cover

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


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


def bench_timed_automata_family(seed: int = _SEED + 1863) -> dict[str, float]:
    return _floats(_finite_blob("timed_automata", bench_timed_automata(seed)))


def bench_parity_game_family(seed: int = _SEED + 1864) -> dict[str, float]:
    return _floats(_finite_blob("parity_game", bench_parity_game(seed)))


def bench_nba_emptiness_family(seed: int = _SEED + 1865) -> dict[str, float]:
    return _floats(_finite_blob("nba_emptiness", bench_nba_emptiness(seed)))


def bench_ctl_mc_family(seed: int = _SEED + 1866) -> dict[str, float]:
    return _floats(_finite_blob("ctl_mc", bench_ctl_mc(seed)))


def bench_bisim_refine_family(seed: int = _SEED + 1867) -> dict[str, float]:
    return _floats(_finite_blob("bisim_refine", bench_bisim_refine(seed)))


def bench_wsts_cover_family(seed: int = _SEED + 1868) -> dict[str, float]:
    return _floats(_finite_blob("wsts_cover", bench_wsts_cover(seed)))
