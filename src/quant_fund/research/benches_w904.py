"""Wave-904 balanced-tree canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.aa_tree import bench_aa_tree
from quant_fund.models.avl_tree import bench_avl_tree
from quant_fund.models.red_black_tree import bench_red_black_tree
from quant_fund.models.scapegoat_tree import bench_scapegoat_tree
from quant_fund.models.splay_tree import bench_splay_tree
from quant_fund.models.treap import bench_treap

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


def bench_avl_tree_family(seed: int = _SEED + 29200) -> dict[str, float]:
    return _finite_blob(bench_avl_tree(seed))


def bench_red_black_tree_family(seed: int = _SEED + 29201) -> dict[str, float]:
    return _finite_blob(bench_red_black_tree(seed))


def bench_splay_tree_family(seed: int = _SEED + 29202) -> dict[str, float]:
    return _finite_blob(bench_splay_tree(seed))


def bench_treap_family(seed: int = _SEED + 29203) -> dict[str, float]:
    return _finite_blob(bench_treap(seed))


def bench_scapegoat_tree_family(seed: int = _SEED + 29204) -> dict[str, float]:
    return _finite_blob(bench_scapegoat_tree(seed))


def bench_aa_tree_family(seed: int = _SEED + 29205) -> dict[str, float]:
    return _finite_blob(bench_aa_tree(seed))
