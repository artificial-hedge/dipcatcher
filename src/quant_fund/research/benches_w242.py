"""Wave-242 adapters: consensus/distributed-2 canon — Multi-Paxos,
EPaxos, Viewstamped, ZAB, SWIM gossip, 2PC/3PC — SYNTHETIC correctness
benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.epaxos import bench_epaxos
from quant_fund.models.multi_paxos import bench_multi_paxos
from quant_fund.models.swim_gossip import bench_swim_gossip
from quant_fund.models.two_three_pc import bench_two_three_pc
from quant_fund.models.viewstamped import bench_viewstamped
from quant_fund.models.zab_protocol import bench_zab_protocol

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


def bench_epaxos_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("epaxos", bench_epaxos(seed=_SEED + 1190)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"epaxos bench failed: {exc}") from exc


def bench_multi_paxos_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("multi_paxos", bench_multi_paxos(seed=_SEED + 1191)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"multi_paxos bench failed: {exc}") from exc


def bench_swim_gossip_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("swim_gossip", bench_swim_gossip(seed=_SEED + 1192)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"swim_gossip bench failed: {exc}") from exc


def bench_two_three_pc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("two_three_pc", bench_two_three_pc(seed=_SEED + 1193)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"two_three_pc bench failed: {exc}") from exc


def bench_viewstamped_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("viewstamped", bench_viewstamped(seed=_SEED + 1194)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"viewstamped bench failed: {exc}") from exc


def bench_zab_protocol_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("zab_protocol", bench_zab_protocol(seed=_SEED + 1195)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"zab_protocol bench failed: {exc}") from exc
