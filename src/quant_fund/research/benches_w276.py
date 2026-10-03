"""Wave-276 distributed-systems-3 benches: mutual exclusion, DHT, quorums."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bully_elect import bench_bully_elect
from quant_fund.models.causal_bcast import bench_causal_bcast
from quant_fund.models.chord_look import bench_chord_look
from quant_fund.models.quorum_rw import bench_quorum_rw
from quant_fund.models.ra_mutex import bench_ra_mutex
from quant_fund.models.token_ring import bench_token_ring

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


def bench_ra_mutex_family(seed: int = _SEED + 1530) -> dict[str, float]:
    return _floats(_finite_blob("ra_mutex", bench_ra_mutex(seed)))


def bench_token_ring_family(seed: int = _SEED + 1531) -> dict[str, float]:
    return _floats(_finite_blob("token_ring", bench_token_ring(seed)))


def bench_bully_elect_family(seed: int = _SEED + 1532) -> dict[str, float]:
    return _floats(_finite_blob("bully_elect", bench_bully_elect(seed)))


def bench_chord_look_family(seed: int = _SEED + 1533) -> dict[str, float]:
    return _floats(_finite_blob("chord_look", bench_chord_look(seed)))


def bench_quorum_rw_family(seed: int = _SEED + 1534) -> dict[str, float]:
    return _floats(_finite_blob("quorum_rw", bench_quorum_rw(seed)))


def bench_causal_bcast_family(seed: int = _SEED + 1535) -> dict[str, float]:
    return _floats(_finite_blob("causal_bcast", bench_causal_bcast(seed)))
