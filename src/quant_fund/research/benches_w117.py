"""Wave-117 adapters: RL canon — GAE, V-trace, TRPO, PPO, DDPG, and
TD3 — each benched on SYNTHETIC environments. Adapters flatten to a
finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ddpg import bench_ddpg
from quant_fund.models.gae import bench_gae
from quant_fund.models.ppo import bench_ppo
from quant_fund.models.td3 import bench_td3
from quant_fund.models.trpo import bench_trpo
from quant_fund.models.vtrace import bench_vtrace

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
                flat[f"{k}_{i}"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_gae_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gae", bench_gae(seed=_SEED + 690)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gae bench failed: {exc}") from exc


def bench_vtrace_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vtrace", bench_vtrace(seed=_SEED + 691)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vtrace bench failed: {exc}") from exc


def bench_trpo_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("trpo", bench_trpo(seed=_SEED + 692)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"trpo bench failed: {exc}") from exc


def bench_ppo_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ppo", bench_ppo(seed=_SEED + 693)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ppo bench failed: {exc}") from exc


def bench_ddpg_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ddpg", bench_ddpg(seed=_SEED + 694)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ddpg bench failed: {exc}") from exc


def bench_td3_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("td3", bench_td3(seed=_SEED + 695)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"td3 bench failed: {exc}") from exc
