"""Wave-966 operator-space canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cb_map import bench_cb_map
from quant_fund.models.complete_contraction import bench_complete_contraction
from quant_fund.models.injective_space import bench_injective_space
from quant_fund.models.noncommutative_lp import bench_noncommutative_lp
from quant_fund.models.oh_emb import bench_oh_emb
from quant_fund.models.operator_space import bench_operator_space

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


def bench_operator_space_family(seed: int = _SEED + 35400) -> dict[str, float]:
    return _finite_blob(bench_operator_space(seed))


def bench_cb_map_family(seed: int = _SEED + 35401) -> dict[str, float]:
    return _finite_blob(bench_cb_map(seed))


def bench_complete_contraction_family(seed: int = _SEED + 35402) -> dict[str, float]:
    return _finite_blob(bench_complete_contraction(seed))


def bench_injective_space_family(seed: int = _SEED + 35403) -> dict[str, float]:
    return _finite_blob(bench_injective_space(seed))


def bench_noncommutative_lp_family(seed: int = _SEED + 35404) -> dict[str, float]:
    return _finite_blob(bench_noncommutative_lp(seed))


def bench_oh_emb_family(seed: int = _SEED + 35405) -> dict[str, float]:
    return _finite_blob(bench_oh_emb(seed))
