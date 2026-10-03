"""Wave-312 distributed-4 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.abd_register import bench_abd_register
from quant_fund.models.bracha_bcast import bench_bracha_bcast
from quant_fund.models.delta_crdt import bench_delta_crdt
from quant_fund.models.hlc_clock import bench_hlc_clock
from quant_fund.models.quorum_weighted import bench_quorum_weighted
from quant_fund.models.raft_log import bench_raft_log
from quant_fund.models.tot_order import bench_tot_order

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


def bench_hlc_clock_family(seed: int = _SEED + 1778) -> dict[str, float]:
    return _floats(_finite_blob("hlc_clock", bench_hlc_clock(seed)))


def bench_delta_crdt_family(seed: int = _SEED + 1779) -> dict[str, float]:
    return _floats(_finite_blob("delta_crdt", bench_delta_crdt(seed)))


def bench_raft_log_family(seed: int = _SEED + 1780) -> dict[str, float]:
    return _floats(_finite_blob("raft_log", bench_raft_log(seed)))


def bench_bracha_bcast_family(seed: int = _SEED + 1781) -> dict[str, float]:
    return _floats(_finite_blob("bracha_bcast", bench_bracha_bcast(seed)))


def bench_tot_order_family(seed: int = _SEED + 1782) -> dict[str, float]:
    return _floats(_finite_blob("tot_order", bench_tot_order(seed)))


def bench_quorum_weighted_family(seed: int = _SEED + 1783) -> dict[str, float]:
    return _floats(_finite_blob("quorum_weighted", bench_quorum_weighted(seed)))


def bench_abd_register_family(seed: int = _SEED + 1784) -> dict[str, float]:
    return _floats(_finite_blob("abd_register", bench_abd_register(seed)))
