"""Wave-909 deque/linked-structure canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.deque_array import bench_deque_array
from quant_fund.models.doubly_linked_list import bench_doubly_linked_list
from quant_fund.models.gap_buffer import bench_gap_buffer
from quant_fund.models.piece_table import bench_piece_table
from quant_fund.models.unrolled_list import bench_unrolled_list
from quant_fund.models.xor_linked_list import bench_xor_linked_list

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


def bench_doubly_linked_list_family(seed: int = _SEED + 29700) -> dict[str, float]:
    return _finite_blob(bench_doubly_linked_list(seed))


def bench_unrolled_list_family(seed: int = _SEED + 29701) -> dict[str, float]:
    return _finite_blob(bench_unrolled_list(seed))


def bench_gap_buffer_family(seed: int = _SEED + 29702) -> dict[str, float]:
    return _finite_blob(bench_gap_buffer(seed))


def bench_piece_table_family(seed: int = _SEED + 29703) -> dict[str, float]:
    return _finite_blob(bench_piece_table(seed))


def bench_deque_array_family(seed: int = _SEED + 29704) -> dict[str, float]:
    return _finite_blob(bench_deque_array(seed))


def bench_xor_linked_list_family(seed: int = _SEED + 29705) -> dict[str, float]:
    return _finite_blob(bench_xor_linked_list(seed))
