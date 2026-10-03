"""Wave-224 adapters: string-algorithm canon — aho_corasick, suffix_automaton,
kmp_search, edit_distance, lz77, bwt_transform —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aho_corasick import bench_aho_corasick
from quant_fund.models.bwt_transform import bench_bwt_transform
from quant_fund.models.edit_distance import bench_edit_distance
from quant_fund.models.kmp_search import bench_kmp_search
from quant_fund.models.lz77 import bench_lz77
from quant_fund.models.suffix_automaton import bench_suffix_automaton

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


def bench_aho_corasick_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("aho_corasick", bench_aho_corasick(seed=_SEED + 1010)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"aho_corasick bench failed: {exc}") from exc


def bench_bwt_transform_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bwt_transform", bench_bwt_transform(seed=_SEED + 1011)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bwt_transform bench failed: {exc}") from exc


def bench_edit_distance_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("edit_distance", bench_edit_distance(seed=_SEED + 1012)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"edit_distance bench failed: {exc}") from exc


def bench_kmp_search_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("kmp_search", bench_kmp_search(seed=_SEED + 1013)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kmp_search bench failed: {exc}") from exc


def bench_lz77_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lz77", bench_lz77(seed=_SEED + 1014)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lz77 bench failed: {exc}") from exc


def bench_suffix_automaton_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("suffix_automaton", bench_suffix_automaton(seed=_SEED + 1015)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"suffix_automaton bench failed: {exc}") from exc
