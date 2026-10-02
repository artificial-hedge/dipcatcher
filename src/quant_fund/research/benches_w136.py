"""Wave-126 adapters: exec-summary attention-efficiency canon — linformer_attn,
performer_attn, linear_attn, sliding_attn, sinkhorn_attn, nystrom_attn —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.linear_attn import bench_linear_attn
from quant_fund.models.linformer_attn import bench_linformer_attn
from quant_fund.models.nystrom_attn import bench_nystrom_attn
from quant_fund.models.performer_attn import bench_performer_attn
from quant_fund.models.sinkhorn_attn import bench_sinkhorn_attn
from quant_fund.models.sliding_attn import bench_sliding_attn

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


def bench_linformer_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("linformer_attn", bench_linformer_attn(seed=_SEED + 804)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"linformer_attn bench failed: {exc}") from exc


def bench_performer_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("performer_attn", bench_performer_attn(seed=_SEED + 805)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"performer_attn bench failed: {exc}") from exc


def bench_linear_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("linear_attn", bench_linear_attn(seed=_SEED + 806)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"linear_attn bench failed: {exc}") from exc


def bench_sliding_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sliding_attn", bench_sliding_attn(seed=_SEED + 807)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sliding_attn bench failed: {exc}") from exc


def bench_sinkhorn_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sinkhorn_attn", bench_sinkhorn_attn(seed=_SEED + 808)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sinkhorn_attn bench failed: {exc}") from exc


def bench_nystrom_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nystrom_attn", bench_nystrom_attn(seed=_SEED + 809)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nystrom_attn bench failed: {exc}") from exc
