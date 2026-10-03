"""Wave-906 persistent-structure canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.finger_tree import bench_finger_tree
from quant_fund.models.persistent_array import bench_persistent_array
from quant_fund.models.pure_queue import bench_pure_queue
from quant_fund.models.rope_string import bench_rope_string
from quant_fund.models.skip_list import bench_skip_list
from quant_fund.models.vlist import bench_vlist

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


def bench_skip_list_family(seed: int = _SEED + 29400) -> dict[str, float]:
    return _finite_blob(bench_skip_list(seed))


def bench_persistent_array_family(seed: int = _SEED + 29401) -> dict[str, float]:
    return _finite_blob(bench_persistent_array(seed))


def bench_finger_tree_family(seed: int = _SEED + 29402) -> dict[str, float]:
    return _finite_blob(bench_finger_tree(seed))


def bench_rope_string_family(seed: int = _SEED + 29403) -> dict[str, float]:
    return _finite_blob(bench_rope_string(seed))


def bench_vlist_family(seed: int = _SEED + 29404) -> dict[str, float]:
    return _finite_blob(bench_vlist(seed))


def bench_pure_queue_family(seed: int = _SEED + 29405) -> dict[str, float]:
    return _finite_blob(bench_pure_queue(seed))
