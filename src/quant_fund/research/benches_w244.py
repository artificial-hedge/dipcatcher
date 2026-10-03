"""Wave-244 adapters: IR canon — inverted index, postings merge,
WAND top-k, MinHash-LSH dedup, n-gram spelling, positional index —
SYNTHETIC correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.inverted_index import bench_inverted_index
from quant_fund.models.lsh_dedup import bench_lsh_dedup
from quant_fund.models.ngram_spell import bench_ngram_spell
from quant_fund.models.positional_index import bench_positional_index
from quant_fund.models.posting_merge import bench_posting_merge
from quant_fund.models.wand_bmw import bench_wand_bmw

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


def bench_inverted_index_family(seed: int = _SEED + 1210) -> dict[str, float]:
    return bench_inverted_index(seed)


def bench_lsh_dedup_family(seed: int = _SEED + 1211) -> dict[str, float]:
    return bench_lsh_dedup(seed)


def bench_ngram_spell_family(seed: int = _SEED + 1212) -> dict[str, float]:
    return bench_ngram_spell(seed)


def bench_positional_index_family(seed: int = _SEED + 1213) -> dict[str, float]:
    return bench_positional_index(seed)


def bench_posting_merge_family(seed: int = _SEED + 1214) -> dict[str, float]:
    return bench_posting_merge(seed)


def bench_wand_bmw_family(seed: int = _SEED + 1215) -> dict[str, float]:
    return bench_wand_bmw(seed)
