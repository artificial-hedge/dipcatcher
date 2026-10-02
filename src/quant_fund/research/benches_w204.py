"""Wave-203 adapters: information-geometry canon — jackson_network,
bcmp_mva, gordon_newell, ctmc_availability, renewal_reward, vacation_queue —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bcmp_mva import bench_bcmp_mva
from quant_fund.models.ctmc_availability import bench_ctmc_availability
from quant_fund.models.gordon_newell import bench_gordon_newell
from quant_fund.models.jackson_network import bench_jackson_network
from quant_fund.models.renewal_reward import bench_renewal_reward
from quant_fund.models.vacation_queue import bench_vacation_queue

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


def bench_renewal_reward_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("renewal_reward", bench_renewal_reward(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"renewal_reward bench failed: {exc}") from exc


def bench_jackson_network_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("jackson_network", bench_jackson_network(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"jackson_network bench failed: {exc}") from exc


def bench_ctmc_availability_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ctmc_availability", bench_ctmc_availability(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ctmc_availability bench failed: {exc}") from exc


def bench_bcmp_mva_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bcmp_mva", bench_bcmp_mva(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bcmp_mva bench failed: {exc}") from exc


def bench_gordon_newell_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gordon_newell", bench_gordon_newell(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gordon_newell bench failed: {exc}") from exc


def bench_vacation_queue_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vacation_queue", bench_vacation_queue(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vacation_queue bench failed: {exc}") from exc
