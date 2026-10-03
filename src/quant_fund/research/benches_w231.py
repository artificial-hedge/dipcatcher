"""Wave-231 adapters: computer-architecture canon — 5-stage pipeline,
LRU cache, branch predictors, Tomasulo OoO, paging/TLB, roofline —
SYNTHETIC correctness benches against invariants and oracles.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.branch_predictor import bench_branch_predictor
from quant_fund.models.cache_sim import bench_cache_sim
from quant_fund.models.cpu_pipeline import bench_cpu_pipeline
from quant_fund.models.paging_sim import bench_paging_sim
from quant_fund.models.roofline_model import bench_roofline_model
from quant_fund.models.tomasulo_sim import bench_tomasulo_sim

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


def bench_branch_predictor_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("branch_predictor", bench_branch_predictor(seed=_SEED + 1080)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"branch_predictor bench failed: {exc}") from exc


def bench_cache_sim_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cache_sim", bench_cache_sim(seed=_SEED + 1081)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cache_sim bench failed: {exc}") from exc


def bench_cpu_pipeline_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cpu_pipeline", bench_cpu_pipeline(seed=_SEED + 1082)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cpu_pipeline bench failed: {exc}") from exc


def bench_paging_sim_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("paging_sim", bench_paging_sim(seed=_SEED + 1083)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"paging_sim bench failed: {exc}") from exc


def bench_roofline_model_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("roofline_model", bench_roofline_model(seed=_SEED + 1084)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"roofline_model bench failed: {exc}") from exc


def bench_tomasulo_sim_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tomasulo_sim", bench_tomasulo_sim(seed=_SEED + 1085)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tomasulo_sim bench failed: {exc}") from exc
