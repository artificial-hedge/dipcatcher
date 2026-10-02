"""Wave-126 adapters: exec-summary LM-components canon — rope_attn,
alibi_attn, swiglu_ffn, rmsnorm_block, moe_router, mup_init —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.alibi_attn import bench_alibi_attn
from quant_fund.models.moe_router import bench_moe_router
from quant_fund.models.mup_init import bench_mup_init
from quant_fund.models.rmsnorm_block import bench_rmsnorm_block
from quant_fund.models.rope_attn import bench_rope_attn
from quant_fund.models.swiglu_ffn import bench_swiglu_ffn

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


def bench_rope_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rope_attn", bench_rope_attn(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rope_attn bench failed: {exc}") from exc


def bench_alibi_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("alibi_attn", bench_alibi_attn(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"alibi_attn bench failed: {exc}") from exc


def bench_swiglu_ffn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("swiglu_ffn", bench_swiglu_ffn(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"swiglu_ffn bench failed: {exc}") from exc


def bench_rmsnorm_block_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rmsnorm_block", bench_rmsnorm_block(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rmsnorm_block bench failed: {exc}") from exc


def bench_moe_router_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("moe_router", bench_moe_router(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"moe_router bench failed: {exc}") from exc


def bench_mup_init_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mup_init", bench_mup_init(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mup_init bench failed: {exc}") from exc
