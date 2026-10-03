"""Wave-122 adapters: execution/microstructure SOTA — exec_rl,
smart_router, order_flow_imbalance, pg_mm, options_flow, dark_pool —
each benched on SYNTHETIC market simulators. Adapters flatten to a
finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dark_pool import bench_dark_pool
from quant_fund.models.exec_rl import bench_exec_rl
from quant_fund.models.options_flow import bench_options_flow
from quant_fund.models.order_flow_imbalance import bench_order_flow_imbalance
from quant_fund.models.pg_mm import bench_pg_mm
from quant_fund.models.smart_router import bench_smart_router

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


def bench_exec_rl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("exec_rl", bench_exec_rl(seed=_SEED + 720)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"exec_rl bench failed: {exc}") from exc


def bench_smart_router_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("smart_router", bench_smart_router(seed=_SEED + 721)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"smart_router bench failed: {exc}") from exc


def bench_order_flow_imbalance_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "order_flow_imbalance",
                bench_order_flow_imbalance(seed=_SEED + 722),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"order_flow_imbalance bench failed: {exc}") from exc


def bench_pg_mm_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pg_mm", bench_pg_mm(seed=_SEED + 723)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pg_mm bench failed: {exc}") from exc


def bench_options_flow_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("options_flow", bench_options_flow(seed=_SEED + 724)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"options_flow bench failed: {exc}") from exc


def bench_dark_pool_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dark_pool", bench_dark_pool(seed=_SEED + 725)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dark_pool bench failed: {exc}") from exc
