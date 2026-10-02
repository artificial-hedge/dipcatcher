"""Wave-126 adapters: exec-summary generative-sequence — vq_vae_ts,
flow_matching_ts, score_sde_ts, consistency_ts, energy_ts, perceiver_ts —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.consistency_ts import bench_consistency_ts
from quant_fund.models.energy_ts import bench_energy_ts
from quant_fund.models.flow_matching_ts import bench_flow_matching_ts
from quant_fund.models.perceiver_ts import bench_perceiver_ts
from quant_fund.models.score_sde_ts import bench_score_sde_ts
from quant_fund.models.vq_vae_ts import bench_vq_vae_ts

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


def bench_vq_vae_ts_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vq_vae_ts", bench_vq_vae_ts(seed=_SEED + 774)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vq_vae_ts bench failed: {exc}") from exc


def bench_flow_matching_ts_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("flow_matching_ts", bench_flow_matching_ts(seed=_SEED + 775)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"flow_matching_ts bench failed: {exc}") from exc


def bench_score_sde_ts_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("score_sde_ts", bench_score_sde_ts(seed=_SEED + 776)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"score_sde_ts bench failed: {exc}") from exc


def bench_consistency_ts_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("consistency_ts", bench_consistency_ts(seed=_SEED + 777)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"consistency_ts bench failed: {exc}") from exc


def bench_energy_ts_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("energy_ts", bench_energy_ts(seed=_SEED + 778)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"energy_ts bench failed: {exc}") from exc


def bench_perceiver_ts_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("perceiver_ts", bench_perceiver_ts(seed=_SEED + 779)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"perceiver_ts bench failed: {exc}") from exc
