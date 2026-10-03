"""Wave-126 adapters: exec-summary agentic canon — react_loop,
toolformer_call, plan_search, reflexion_retry, multi_agent_pipeline, judge_pairwise —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.judge_pairwise import bench_judge_pairwise
from quant_fund.models.multi_agent_pipeline import bench_multi_agent_pipeline
from quant_fund.models.plan_search import bench_plan_search
from quant_fund.models.react_loop import bench_react_loop
from quant_fund.models.reflexion_retry import bench_reflexion_retry
from quant_fund.models.toolformer_call import bench_toolformer_call

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


def bench_react_loop_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("react_loop", bench_react_loop(seed=_SEED + 894)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"react_loop bench failed: {exc}") from exc


def bench_toolformer_call_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("toolformer_call", bench_toolformer_call(seed=_SEED + 895)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"toolformer_call bench failed: {exc}") from exc


def bench_plan_search_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("plan_search", bench_plan_search(seed=_SEED + 896)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"plan_search bench failed: {exc}") from exc


def bench_reflexion_retry_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("reflexion_retry", bench_reflexion_retry(seed=_SEED + 897)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"reflexion_retry bench failed: {exc}") from exc


def bench_multi_agent_pipeline_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("multi_agent_pipeline", bench_multi_agent_pipeline(seed=_SEED + 898))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"multi_agent_pipeline bench failed: {exc}") from exc


def bench_judge_pairwise_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("judge_pairwise", bench_judge_pairwise(seed=_SEED + 899)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"judge_pairwise bench failed: {exc}") from exc
