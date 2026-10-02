"""Wave-119 adapters: MARL canon — VDN, QMIX, COMA, MADDPG, MAPPO,
and mean-field Q — each benched on SYNTHETIC cooperative games.
Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.coma import bench_coma
from quant_fund.models.maddpg import bench_maddpg
from quant_fund.models.mappo import bench_mappo
from quant_fund.models.mf_q import bench_mf_q
from quant_fund.models.qmix import bench_qmix
from quant_fund.models.vdn import bench_vdn

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


def bench_vdn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vdn", bench_vdn(seed=_SEED + 702)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vdn bench failed: {exc}") from exc


def bench_qmix_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qmix", bench_qmix(seed=_SEED + 703)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qmix bench failed: {exc}") from exc


def bench_coma_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("coma", bench_coma(seed=_SEED + 704)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"coma bench failed: {exc}") from exc


def bench_maddpg_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("maddpg", bench_maddpg(seed=_SEED + 705)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"maddpg bench failed: {exc}") from exc


def bench_mappo_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mappo", bench_mappo(seed=_SEED + 706)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mappo bench failed: {exc}") from exc


def bench_mf_q_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mf_q", bench_mf_q(seed=_SEED + 707)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mf_q bench failed: {exc}") from exc
