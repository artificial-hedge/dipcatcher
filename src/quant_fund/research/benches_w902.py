"""Wave-902 trie/string-index canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.crit_bit_tree import bench_crit_bit_tree
from quant_fund.models.patricia_trie import bench_patricia_trie
from quant_fund.models.radix_trie import bench_radix_trie
from quant_fund.models.suffix_trie import bench_suffix_trie
from quant_fund.models.ternary_trie import bench_ternary_trie
from quant_fund.models.trie import bench_trie

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_trie_family(seed: int = _SEED + 29000) -> dict[str, float]:
    return _finite_blob(bench_trie(seed))


def bench_patricia_trie_family(seed: int = _SEED + 29001) -> dict[str, float]:
    return _finite_blob(bench_patricia_trie(seed))


def bench_suffix_trie_family(seed: int = _SEED + 29002) -> dict[str, float]:
    return _finite_blob(bench_suffix_trie(seed))


def bench_ternary_trie_family(seed: int = _SEED + 29003) -> dict[str, float]:
    return _finite_blob(bench_ternary_trie(seed))


def bench_radix_trie_family(seed: int = _SEED + 29004) -> dict[str, float]:
    return _finite_blob(bench_radix_trie(seed))


def bench_crit_bit_tree_family(seed: int = _SEED + 29005) -> dict[str, float]:
    return _finite_blob(bench_crit_bit_tree(seed))
