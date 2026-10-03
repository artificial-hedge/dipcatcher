"""Wave-126 adapters: exec-summary alignment canon — reward_model,
dpo_train, ipo_train, kto_train, grpo_train, rlhf_ppo —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dpo_train import bench_dpo_train
from quant_fund.models.grpo_train import bench_grpo_train
from quant_fund.models.ipo_train import bench_ipo_train
from quant_fund.models.kto_train import bench_kto_train
from quant_fund.models.reward_model import bench_reward_model
from quant_fund.models.rlhf_ppo import bench_rlhf_ppo

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


def bench_reward_model_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("reward_model", bench_reward_model(seed=_SEED + 870)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"reward_model bench failed: {exc}") from exc


def bench_dpo_train_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dpo_train", bench_dpo_train(seed=_SEED + 871)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dpo_train bench failed: {exc}") from exc


def bench_ipo_train_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ipo_train", bench_ipo_train(seed=_SEED + 872)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ipo_train bench failed: {exc}") from exc


def bench_kto_train_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("kto_train", bench_kto_train(seed=_SEED + 873)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kto_train bench failed: {exc}") from exc


def bench_grpo_train_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("grpo_train", bench_grpo_train(seed=_SEED + 874)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"grpo_train bench failed: {exc}") from exc


def bench_rlhf_ppo_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rlhf_ppo", bench_rlhf_ppo(seed=_SEED + 875)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rlhf_ppo bench failed: {exc}") from exc
