"""Wave-223 adapters: distributed-systems canon — paxos, raft_election,
vector_clock, consistent_hash, gossip_epidemic, pbft_lite —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.consistent_hash import bench_consistent_hash
from quant_fund.models.gossip_epidemic import bench_gossip_epidemic
from quant_fund.models.paxos import bench_paxos
from quant_fund.models.pbft_lite import bench_pbft_lite
from quant_fund.models.raft_election import bench_raft_election
from quant_fund.models.vector_clock import bench_vector_clock

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


def bench_consistent_hash_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("consistent_hash", bench_consistent_hash(seed=_SEED + 1000)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"consistent_hash bench failed: {exc}") from exc


def bench_gossip_epidemic_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gossip_epidemic", bench_gossip_epidemic(seed=_SEED + 1001)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gossip_epidemic bench failed: {exc}") from exc


def bench_paxos_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("paxos", bench_paxos(seed=_SEED + 1002)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"paxos bench failed: {exc}") from exc


def bench_pbft_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pbft_lite", bench_pbft_lite(seed=_SEED + 1003)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pbft_lite bench failed: {exc}") from exc


def bench_raft_election_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("raft_election", bench_raft_election(seed=_SEED + 1004)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"raft_election bench failed: {exc}") from exc


def bench_vector_clock_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vector_clock", bench_vector_clock(seed=_SEED + 1005)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vector_clock bench failed: {exc}") from exc
