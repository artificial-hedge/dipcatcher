"""Wave-203 adapters: information-geometry canon — ucb_bound,
mw_hedge, egreedy_decay, pi_contraction, td_rate, qlearn_rate —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.egreedy_decay import bench_egreedy_decay
from quant_fund.models.mw_hedge import bench_mw_hedge
from quant_fund.models.pi_contraction import bench_pi_contraction
from quant_fund.models.qlearn_rate import bench_qlearn_rate
from quant_fund.models.td_rate import bench_td_rate
from quant_fund.models.ucb_bound import bench_ucb_bound

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


def bench_td_rate_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("td_rate", bench_td_rate(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"td_rate bench failed: {exc}") from exc


def bench_ucb_bound_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ucb_bound", bench_ucb_bound(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ucb_bound bench failed: {exc}") from exc


def bench_pi_contraction_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pi_contraction", bench_pi_contraction(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pi_contraction bench failed: {exc}") from exc


def bench_mw_hedge_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mw_hedge", bench_mw_hedge(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mw_hedge bench failed: {exc}") from exc


def bench_egreedy_decay_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("egreedy_decay", bench_egreedy_decay(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"egreedy_decay bench failed: {exc}") from exc


def bench_qlearn_rate_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qlearn_rate", bench_qlearn_rate(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qlearn_rate bench failed: {exc}") from exc
