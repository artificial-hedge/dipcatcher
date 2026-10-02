"""Wave-126 adapters: exec-summary parameter-efficient finetuning canon — lora_ft,
qlora_nf4, dora_weight, prompt_tuning, prefix_tuning, task_vector_merge —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dora_weight import bench_dora_weight
from quant_fund.models.lora_ft import bench_lora_ft
from quant_fund.models.prefix_tuning import bench_prefix_tuning
from quant_fund.models.prompt_tuning import bench_prompt_tuning
from quant_fund.models.qlora_nf4 import bench_qlora_nf4
from quant_fund.models.task_vector_merge import bench_task_vector_merge

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


def bench_lora_ft_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lora_ft", bench_lora_ft(seed=_SEED + 846)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lora_ft bench failed: {exc}") from exc


def bench_qlora_nf4_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qlora_nf4", bench_qlora_nf4(seed=_SEED + 847)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qlora_nf4 bench failed: {exc}") from exc


def bench_dora_weight_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dora_weight", bench_dora_weight(seed=_SEED + 848)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dora_weight bench failed: {exc}") from exc


def bench_prompt_tuning_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("prompt_tuning", bench_prompt_tuning(seed=_SEED + 849)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"prompt_tuning bench failed: {exc}") from exc


def bench_prefix_tuning_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("prefix_tuning", bench_prefix_tuning(seed=_SEED + 850)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"prefix_tuning bench failed: {exc}") from exc


def bench_task_vector_merge_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("task_vector_merge", bench_task_vector_merge(seed=_SEED + 851)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"task_vector_merge bench failed: {exc}") from exc
