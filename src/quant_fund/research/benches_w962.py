"""Wave-962 von-Neumann-algebra canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.double_commutant import bench_double_commutant
from quant_fund.models.jones_index import bench_jones_index
from quant_fund.models.normal_state import bench_normal_state
from quant_fund.models.predual_space import bench_predual_space
from quant_fund.models.tomita_takesaki import bench_tomita_takesaki
from quant_fund.models.von_neumann_alg import bench_von_neumann_alg

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


def bench_von_neumann_alg_family(seed: int = _SEED + 35000) -> dict[str, float]:
    return _finite_blob(bench_von_neumann_alg(seed))


def bench_double_commutant_family(seed: int = _SEED + 35001) -> dict[str, float]:
    return _finite_blob(bench_double_commutant(seed))


def bench_predual_space_family(seed: int = _SEED + 35002) -> dict[str, float]:
    return _finite_blob(bench_predual_space(seed))


def bench_normal_state_family(seed: int = _SEED + 35003) -> dict[str, float]:
    return _finite_blob(bench_normal_state(seed))


def bench_tomita_takesaki_family(seed: int = _SEED + 35004) -> dict[str, float]:
    return _finite_blob(bench_tomita_takesaki(seed))


def bench_jones_index_family(seed: int = _SEED + 35005) -> dict[str, float]:
    return _finite_blob(bench_jones_index(seed))
