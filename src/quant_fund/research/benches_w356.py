"""Wave-356 stochastic-processes-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gambler_ruin import bench_gambler_ruin
from quant_fund.models.markov_chain import bench_markov_chain
from quant_fund.models.markov_hitting import bench_markov_hitting
from quant_fund.models.martingale_check import bench_martingale_check
from quant_fund.models.poisson_process import bench_poisson_process
from quant_fund.models.stopping_time import bench_stopping_time

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


def bench_markov_chain_family(seed: int = _SEED + 2043) -> dict[str, float]:
    return _floats(_finite_blob("markov_chain", bench_markov_chain(seed)))


def bench_martingale_check_family(seed: int = _SEED + 2044) -> dict[str, float]:
    return _floats(_finite_blob("martingale_check", bench_martingale_check(seed)))


def bench_poisson_process_family(seed: int = _SEED + 2045) -> dict[str, float]:
    return _floats(_finite_blob("poisson_process", bench_poisson_process(seed)))


def bench_gambler_ruin_family(seed: int = _SEED + 2046) -> dict[str, float]:
    return _floats(_finite_blob("gambler_ruin", bench_gambler_ruin(seed)))


def bench_stopping_time_family(seed: int = _SEED + 2047) -> dict[str, float]:
    return _floats(_finite_blob("stopping_time", bench_stopping_time(seed)))


def bench_markov_hitting_family(seed: int = _SEED + 2048) -> dict[str, float]:
    return _floats(_finite_blob("markov_hitting", bench_markov_hitting(seed)))
