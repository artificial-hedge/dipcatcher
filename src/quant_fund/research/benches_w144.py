"""Wave-126 adapters: exec-summary inference-engine canon — speculative_decoding,
paged_kv_cache, flash_attn, gqa_attn, sliding_window_cache, ring_attn —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.flash_attn import bench_flash_attn
from quant_fund.models.gqa_attn import bench_gqa_attn
from quant_fund.models.paged_kv_cache import bench_paged_kv_cache
from quant_fund.models.ring_attn import bench_ring_attn
from quant_fund.models.sliding_window_cache import bench_sliding_window_cache
from quant_fund.models.speculative_decoding import bench_speculative_decoding

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


def bench_speculative_decoding_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("speculative_decoding", bench_speculative_decoding(seed=_SEED + 852))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"speculative_decoding bench failed: {exc}") from exc


def bench_paged_kv_cache_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("paged_kv_cache", bench_paged_kv_cache(seed=_SEED + 853)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"paged_kv_cache bench failed: {exc}") from exc


def bench_flash_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("flash_attn", bench_flash_attn(seed=_SEED + 854)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"flash_attn bench failed: {exc}") from exc


def bench_gqa_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gqa_attn", bench_gqa_attn(seed=_SEED + 855)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gqa_attn bench failed: {exc}") from exc


def bench_sliding_window_cache_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("sliding_window_cache", bench_sliding_window_cache(seed=_SEED + 856))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sliding_window_cache bench failed: {exc}") from exc


def bench_ring_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ring_attn", bench_ring_attn(seed=_SEED + 857)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ring_attn bench failed: {exc}") from exc
