"""Wave-234 adapters: database-internals canon — B+tree, WAL recovery,
join algorithms, cost-based planner, MVCC, LSM compaction — SYNTHETIC
correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.btree_index import bench_btree_index
from quant_fund.models.join_algos import bench_join_algos
from quant_fund.models.lsm_tree import bench_lsm_tree
from quant_fund.models.mvcc_isolation import bench_mvcc_isolation
from quant_fund.models.query_planner import bench_query_planner
from quant_fund.models.wal_recovery import bench_wal_recovery

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


def bench_btree_index_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("btree_index", bench_btree_index(seed=_SEED + 1110)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"btree_index bench failed: {exc}") from exc


def bench_join_algos_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("join_algos", bench_join_algos(seed=_SEED + 1111)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"join_algos bench failed: {exc}") from exc


def bench_lsm_tree_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lsm_tree", bench_lsm_tree(seed=_SEED + 1112)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lsm_tree bench failed: {exc}") from exc


def bench_mvcc_isolation_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mvcc_isolation", bench_mvcc_isolation(seed=_SEED + 1113)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mvcc_isolation bench failed: {exc}") from exc


def bench_query_planner_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("query_planner", bench_query_planner(seed=_SEED + 1114)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"query_planner bench failed: {exc}") from exc


def bench_wal_recovery_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("wal_recovery", bench_wal_recovery(seed=_SEED + 1115)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"wal_recovery bench failed: {exc}") from exc
